import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, pretrained=False):
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=pretrained)

    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output).squeeze(-1).squeeze(-1)
        return output

class ResNet50(nn.Module):
    def __init__(self, pretrained = False):
        super(ResNet50, self).__init__()
        self.net = models.resnet50(pretrained=pretrained)
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output).squeeze(-1).squeeze(-1)
        return output
    
class ResNet101(nn.Module):
    def __init__(self, pretrained = False):
        super(ResNet101, self).__init__()
        self.net = models.resnet101(pretrained=pretrained)
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output).squeeze(-1).squeeze(-1)
        return output