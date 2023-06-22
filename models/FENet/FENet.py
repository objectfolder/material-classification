import sys
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Material_Classification/code/models/FENet/VGGish')
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Material_Classification/code/models/FENet/FAPool')
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Material_Classification/code/models/FENet/')
import librosa
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
import encoding

from .ResNet import ResNet18, ResNet50, ResNet101
from .DenseNet import DenseNet201
from .VGG import VGG19
from .VGGish.vggish import VGGish
from .VGGish.audioset import vggish_input, vggish_postprocess
from .FAPool.FAP import FAP

backbone_dict = {
    'vision': {'resnet18': ResNet18, 'resnet50': ResNet50, 'resnet101': ResNet101},
    'touch': {'resnet18': ResNet18, 'resnet50': ResNet50, 'resnet101': ResNet101},
    'audio': {'vggish': VGGish}
}

feature_dim_dict = {
    'resnet18': 512, 
    'resnet50': 2048, 'resnet101': 2048, 'vggish': 128
}

class FENet(nn.Module):
    def __init__(self, args, cfg):
        super(FENet, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_vision = 'vision' in self.args.modality_list
        self.use_touch = 'touch' in self.args.modality_list
        self.use_audio = 'audio' in self.args.modality_list
        self.dim = self.cfg.dim
        
        self.feature_dim = 0
        if self.use_vision:
            self.vision_backbone, feature_dim = self.build_backbone('vision')
            self.pool_vision = nn.Sequential(
                nn.AvgPool2d(7),
                encoding.nn.View(-1, feature_dim),
                nn.Linear(feature_dim, self.dim*3),
                nn.BatchNorm1d(self.dim*3),
            )   
            self.UP_vision = nn.ConvTranspose2d(feature_dim, 512, 3, 2, groups=512)
            self.conv_before_mfs_vision = nn.Sequential(
                nn.Conv2d(512, 3, kernel_size=1),
                nn.ReLU(),
                nn.BatchNorm2d(3),
            )
            self.mfs_vision = FAP(self.cfg, D=1, K=self.dim)
            self.fc_vision = nn.Sequential(
                encoding.nn.Normalize(),
                nn.Linear((self.dim*3)*(self.dim*3), 128),
                encoding.nn.Normalize(),
            )
            self.mlp_vision = nn.Sequential(
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.Linear(64, 64),
                nn.LeakyReLU(),
            )
            self.feature_dim += 64
        if self.use_touch:
            self.touch_backbone, feature_dim = self.build_backbone('touch')
            self.mlp_touch = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
            )
            self.feature_dim += 64

        if self.use_audio:
            self.audio_backbone, feature_dim = self.build_backbone('audio')
            self.mlp_audio = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
            )
            self.feature_dim += 64

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.Dropout(0.25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.25),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 7),
        )
        self.loss = nn.CrossEntropyLoss()

    def build_backbone(self, modality):
        backbone_name = self.cfg.get(f'{modality}_backbone', False)
        pretrained = True
        backbone = backbone_dict[modality][backbone_name](pretrained)
        feature_dim = feature_dim_dict[backbone_name]
        return backbone, feature_dim
    
    def encode_vision(self, x):
        x = self.vision_backbone.net.conv1(x)
        x = self.vision_backbone.net.bn1(x)
        x = self.vision_backbone.net.relu(x)
        x = self.vision_backbone.net.maxpool(x)
        x = self.vision_backbone.net.layer1(x)
        x = self.vision_backbone.net.layer2(x)
        x = self.vision_backbone.net.layer3(x)
        vision_feature = self.vision_backbone.net.layer4(x)
        up = self.UP_vision(vision_feature) # (bs, 512, 15, 15)
        c = self.conv_before_mfs_vision(up) # (bs, 3, 15, 15)
        c0 = c[:,0,:,:].unsqueeze_(1) # (bs, 1, 15, 15)
        c1 = c[:,1,:,:].unsqueeze_(1) # (bs, 1, 15, 15)
        c2 = c[:,2,:,:].unsqueeze_(1) # (bs, 1, 15, 15)
        fracdim0 = self.mfs_vision(c0).squeeze_(-1).squeeze_(-1) # (bs, 15)
        fracdim1 = self.mfs_vision(c1).squeeze_(-1).squeeze_(-1) # (bs, 15)
        fracdim2 = self.mfs_vision(c2).squeeze_(-1).squeeze_(-1) # (bs, 15)
        x1 = self.pool_vision(vision_feature) # (bs, 45) # avg pooling       
        x2 = torch.cat((fracdim0,fracdim1,fracdim2),1) # (bs, 45)
        x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1)) # (bs, 45, 45) # Bilinear Models
        x3 = x1*x2.unsqueeze(-1)# (bs, 45, 45) #Bilinear Models
        x3 = x3.view(-1,x1.size(-1)*x2.size(1)) #Bilinear Models
        vision_feature = self.fc_vision(x3) # (bs, 128)
        vision_feature = self.mlp_vision(vision_feature) # (bs, 32)
        return vision_feature
    
    def encode_touch(self, x):
        x = self.touch_backbone.net.conv1(x)
        x = self.touch_backbone.net.bn1(x)
        x = self.touch_backbone.net.relu(x)
        x = self.touch_backbone.net.maxpool(x)
        x = self.touch_backbone.net.layer1(x)
        x = self.touch_backbone.net.layer2(x)
        x = self.touch_backbone.net.layer3(x)
        touch_feature = self.touch_backbone.net.layer4(x)
        touch_feature = self.touch_backbone.net.avgpool(touch_feature).squeeze(-1).squeeze(-1)
        touch_feature = self.mlp_touch(touch_feature) # (bs, 32)
        return touch_feature
    
    def encode_audio(self, x):
        audio_feature = self.audio_backbone.features(x).permute(0, 2, 3, 1).contiguous() # (bs, dim, 6, 4)
        audio_feature = audio_feature.view(audio_feature.size(0), -1)
        audio_feature = self.audio_backbone.fc(audio_feature)
        audio_feature = self.mlp_audio(audio_feature) # (bs, 32)
        return audio_feature
    
    def export_module_checkpoint(self):
        return {}
    
    def forward(self, batch, calc_loss=True):
        output = {}
        feature = []
        if self.use_vision:
            vision_feature = self.encode_vision(batch['visual_image'].cuda()) # (bs, 128)
            feature.append(vision_feature)
        if self.use_touch:
            touch_feature = self.encode_touch(batch['tactile_image'].cuda()) # (bs, 128)
            feature.append(touch_feature)
        if self.use_audio:
            audio_feature = self.encode_audio(batch['audio_spectrogram'].cuda()) # (bs, 128)
            feature.append(audio_feature)
        feature = torch.cat(feature, 1) # (bs, m*128)
        pred = self.classifier(feature) # (bs, 7)
        output['pred'] = pred
        if calc_loss:
            target = batch['label'].cuda()
            output['loss'] = self.loss(pred, target)
        
        return output
            
            

        