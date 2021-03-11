#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  @Email:  guo@locationmind.com
  @Copyright: guo-zhiling
  @License: MIT
"""
import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from blockunits import ConvBlock

class Bottleneck(nn.Module):
    # jump three blocks
    expansion = 2
    def __init__(self, in_ch, out_ch, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, self.expansion*out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*out_ch)
        
        # shortcut is to match dimension 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != self.expansion*out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, self.expansion*out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_ch)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    """
    https://medium.com/@jonathan_hui/understanding-feature-pyramid-networks-for-object-detection-fpn-45b227b9106c
    """
    def __init__(self, 
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 zks=3,):
        super(FPN, self).__init__()
        block = Bottleneck
        self.base_kernel = base_kernel
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        num_blocks = [2, 2, 2, 2, 2]

        # Bottom-up layers
        self.layer1 = self._make_layer(block, nb_channel, kernels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, kernels[0] * block.expansion, kernels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, kernels[1] * block.expansion, kernels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, kernels[2] * block.expansion, kernels[3], num_blocks[3], stride=2)
        self.layer5 = self._make_layer(block, kernels[3] * block.expansion, kernels[4], num_blocks[4], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(kernels[4] * block.expansion, kernels[2], kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Lateral layers
        self.latlayer4 = nn.Conv2d(kernels[3] * block.expansion, kernels[2], kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( kernels[2] * block.expansion, kernels[2], kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( kernels[1] * block.expansion, kernels[2], kernel_size=1, stride=1, padding=0)
        
        # Smooth layers
        self.smooth5 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, stride=1, padding=1)
        self.smooth4 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(kernels[2], kernels[2], kernel_size=3, stride=1, padding=1)

        # Side conv
        self.sideconv = ConvBlock(kernels[2], nb_class, is_bn = True)
        
        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(4*nb_class, nb_class, 3, padding=1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)
        
    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(in_ch, out_ch, stride))
            in_ch = out_ch * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='nearest') + y

    def forward(self, x):
        # Bottom-up
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)
        # Top-down
        m5 = self.toplayer(c5)
        m4 = self._upsample_add(m5, self.latlayer4(c4))
        m3 = self._upsample_add(m4, self.latlayer3(c3))
        m2 = self._upsample_add(m3, self.latlayer2(c2))
        # Smooth
        p5 = self.smooth5(m5)
        p4 = self.smooth4(m4)
        p3 = self.smooth3(m3)
        p2 = self.smooth2(m2)
        # Output
        _out5 = self.sideconv(p5)
        out5 = F.interpolate(_out5, scale_factor=16, mode='bilinear', align_corners=False)
        _out4 = self.sideconv(p4)
        out4 = F.interpolate(_out4, scale_factor=8, mode='bilinear', align_corners=False)
        _out3 = self.sideconv(p3)
        out3 = F.interpolate(_out3, scale_factor=4, mode='bilinear', align_corners=False)
        _out2 = self.sideconv(p2)
        out2 = F.interpolate(_out2, scale_factor=2, mode='bilinear', align_corners=False)        
        
        cat = torch.cat([out5, out4, out3, out2], dim=1)
        
        return self.outconv1(cat)
           
if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = torch.FloatTensor(
            np.random.random((1, nb_channel, 224, 224)))
    
    generator = FPN(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("FPN->:")
    print(" Network output ", gen_y.shape)