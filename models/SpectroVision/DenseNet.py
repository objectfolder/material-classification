import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models

class DenseNet201(nn.Module):
    def __init__(self, pretrained = False):
        super(DenseNet201, self).__init__()
        self.net = models.densenet201(pretrained=pretrained)
 
    def forward(self, input):
        features = self.net.features(input)
        out = F.relu(features, )
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out