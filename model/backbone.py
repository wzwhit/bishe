import torch
import torch.nn as nn
from torchvision import models
from model.resnet_block import Bottleneck

class resnet18(nn.Module):
    def __init__(self, block, pretrained=True):
        super(resnet18, self).__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self._make_layer(block, 128, 256, 2, stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 256, 512, 2, stride=1, dilation=2)
        # self.layer3 = nn.Sequential(nn.Conv2d(48,32,kernel_size=1,stride=1,bias=False),
        #                             nn.BatchNorm2d(32),
        #                             nn.ReLU())

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilation=2):
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, dilation=dilation, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(planes, planes, dilation=dilation, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(input))))   # 1/4, 64
        feature1 = self.layer1(x)   # 1/4, 64
        feature2 = self.layer2(feature1)    # 1/8, 128
        feature3 = self.layer3(feature2)    # 1/8, 256
        feature4 = self.layer4(feature3)    # 1/8, 512
        # print(feature4.shape)

        return feature4

if __name__ == '__main__':
    #
    model_18 = resnet18(Bottleneck, pretrained=True)
    x = torch.rand(1, 3, 256, 256)

    y_18 = model_18(x)
