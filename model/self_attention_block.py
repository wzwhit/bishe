import torch
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
import functools

class _SelfAttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.key_channels),
            nn.LeakyReLU(negative_slope=0.01,inplace=True)
        )
        self.f_query = self.f_key
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)  # batchsize*hw*c
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)  # batchsize*hw*c
        key = self.f_key(x).view(batch_size, self.key_channels, -1) # batchsize*c*hw

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)    # batchsize*hw*hw

        context = torch.matmul(sim_map, value)  # batchsize*hw*c
        context = context.permute(0, 2, 1).contiguous() # batchsize*c*hw
        context = context.view(batch_size, self.value_channels, *x.size()[2:])# batchsize*c*h*w
        context = self.W(context)
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)

        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)

class InterlacedSparseSelfAttention(nn.Module):
    def __init__(self,  in_channels, H, W, P_h, P_w):
        super(InterlacedSparseSelfAttention, self).__init__()
        self.in_channels = in_channels
        self.out_channels = self.in_channels
        self.H, self.W, self.P_h, self.P_w = H, W, P_h, P_w
        self.Q_h, self.Q_w = H // P_h, W // P_w
        self.long_SelfAttention = SelfAttentionBlock2D(self.in_channels, self.in_channels//2, self.in_channels//2, self.out_channels, 1)
        self.short_SelfAttention = SelfAttentionBlock2D(self.in_channels, self.in_channels//2, self.in_channels//2, self.out_channels, 1)

    def forward(self, input):
        N, C, H, W = input.size()
        assert C == self.in_channels and H == self.H and W == self.W, 'C != self.in_channels or H != self.H or W != self.W'

        x = input.reshape(N, self.in_channels, self.Q_h, self.P_h, self.Q_w, self.P_w)

        # Long-range Attention
        x = x.permute(0, 3, 5, 1, 2, 4)
        x = x.reshape(N * self.P_h * self.P_w, self.in_channels, self.Q_h, self.Q_w)
        x = self.long_SelfAttention(x)
        x = x.reshape(N, self.P_h, self.P_w, self.in_channels, self.Q_h, self.Q_w)

        # Short-range Attention
        x = x.permute(0, 4, 5, 3, 1, 2)
        x = x.reshape(N * self.Q_h * self.Q_w, self.in_channels, self.P_h, self.P_w)
        x = self.short_SelfAttention(x)
        x = x.reshape(N, self.Q_h, self.Q_w, self.in_channels, self.P_h, self.P_w)

        return x.permute(0, 3, 1, 4, 2, 5).reshape(N, self.in_channels, self.H, self.W)


if __name__ == '__main__':
    #
    isa = InterlacedSparseSelfAttention(in_channels=3, H=256, W=256, P_h=16, P_w=16)
    x = torch.rand(1, 3, 256, 256)

    y_18 = isa(x)
    print(y_18)