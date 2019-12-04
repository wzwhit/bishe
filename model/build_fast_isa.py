import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import numpy as np
from torch.autograd import Variable
import functools
import time
from model.resnet_block import Bottleneck
from model.backbone import resnet18
from model.self_attention_block import InterlacedSparseSelfAttention


class ISA(nn.Module):
    def __init__(self, num_classes=20, out_channels=256, isa_H=256, isa_W=256, isa_P_h=16, isa_P_w=16):
        super(ISA, self).__init__()
        resnet_out_channels = 512
        isa_in_channels = resnet_out_channels // 4  # 128

        self.backbone = resnet18(Bottleneck, pretrained=True)
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(resnet_out_channels, isa_in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(isa_in_channels),
            nn.ReLU())
        self.isa_module = InterlacedSparseSelfAttention(in_channels=isa_in_channels,
                                                        H=isa_H, W=isa_W, P_h=isa_P_h, P_w=isa_P_w)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(resnet_out_channels+isa_in_channels, out_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
            )

        self.interpolate_conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

    def forward(self, input):
        # resnet18, output_stride=8
        x = self.backbone(input)
        # reduce input channels of isa module
        output = self.reduction_conv(x)
        # isa module
        output = self.isa_module(output)
        # feature fuse
        output = self.bottleneck(torch.cat([x, output], 1))

        # upsampling
        output = torch.nn.functional.interpolate(output, scale_factor=8, mode='bilinear', align_corners=True)
        output = self.interpolate_conv(output)

        return output


if __name__ == '__main__':
    #
    model = ISA(num_classes=20, out_channels=256, isa_H=196, isa_W=64, isa_P_h=14, isa_P_w=8)
    model.cuda()
    t = 0
    for i in range(1000):
        x = torch.rand(1, 3, 1568, 512)
        x = x.cuda()
        start = time.time()
        output = model(x)
        end = time.time()
        t += end - start
    print(t/1000)
