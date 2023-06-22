import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models

class VGG19(nn.Module):
    def __init__(self, pretrained = False):
        super(VGG19, self).__init__()
        self.net = models.vgg19(pretrained=pretrained)
 
    def forward(self, input):
        features = self.net.features(input)
        out = self.net.avgpool(features)
        out = torch.flatten(out, 1)
        return out