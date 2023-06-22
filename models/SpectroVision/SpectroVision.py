import sys
sys.path.append('/viscam/u/yimingdou/ObjectFolder-Benchmark/benchmarks/Material_Classification/code/models/SpectroVision/VGGish')
from turtle import forward
import librosa
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models

from .ResNet import ResNet18, ResNet50, ResNet101
from .DenseNet import DenseNet201
from .VGG import VGG19
from .VGGish.vggish import VGGish
from .VGGish.audioset import vggish_input, vggish_postprocess

backbone_dict = {
    'vision': {'densenet201': DenseNet201, 'resnet18': ResNet18, 'resnet50': ResNet50, 'resnet101': ResNet101},
    'touch': {'densenet201': DenseNet201, 'resnet18': ResNet18, 'resnet50': ResNet50, 'resnet101': ResNet101},
    'audio': {'vggish': VGGish}
}
feature_dim_dict = {
    'densenet201': 1920, 'resnet18': 512, 
    'resnet50': 2048, 'resnet101': 2048, 'vggish': 128
}

class SpectroVision(nn.Module):
    def __init__(self, args, cfg):
        super(SpectroVision, self).__init__()
        self.args = args
        self.cfg = cfg
        self.use_vision = 'vision' in self.args.modality_list
        self.use_touch = 'touch' in self.args.modality_list
        self.use_audio = 'audio' in self.args.modality_list
        classifier_input_dim = 0
        if self.use_vision:
            self.vision_backbone, feature_dim = self.build_backbone('vision')
            self.vision_mlp = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 32),
            )
            classifier_input_dim += 32
            
        if self.use_touch:
            self.touch_backbone, feature_dim = self.build_backbone('touch')
            self.touch_mlp = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 32),
            )
            classifier_input_dim += 32
            
        if self.use_audio:
            self.audio_backbone, feature_dim = self.build_backbone('audio')
            self.audio_mlp = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.Dropout(0.25),
                nn.Linear(128, 32),
            )
            classifier_input_dim += 32
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
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
    
    def export_module_checkpoint(self):
        output={'classifier':self.classifier.state_dict()}
        if self.use_vision:
            output['vision_backbone']=self.vision_backbone.state_dict()
            output['vision_mlp']=self.vision_mlp.state_dict()
        if self.use_audio:
            output['audio_backbone']=self.audio_backbone.state_dict()
            output['audio_mlp']=self.audio_mlp.state_dict()
        if self.use_touch:
            output['touch_backbone']=self.touch_backbone.state_dict()
            output['touch_mlp']=self.touch_mlp.state_dict()
        
        
        return output
    
    def forward(self, batch, calc_loss=True):
        output = {}
        features=[]
        if self.use_vision:
            vision_feature = self.vision_backbone(batch['visual_image'].cuda())
            vision_feature = self.vision_mlp(vision_feature)
            features.append(vision_feature)
        if self.use_touch:
            touch_feature = self.touch_backbone(batch['tactile_image'].cuda())
            touch_feature = self.touch_mlp(touch_feature)
            features.append(touch_feature)
        if self.use_audio:
            audio_feature = self.audio_backbone(batch['audio_spectrogram'].cuda())
            audio_feature = self.audio_mlp(audio_feature)
            features.append(audio_feature)
        features = torch.cat(features,dim=1)
        pred = self.classifier(features)
        output['pred'] = pred
        if calc_loss:
            target = batch['label'].cuda()
            output['loss'] = self.loss(pred, target)
        return output
        