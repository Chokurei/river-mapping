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
from blockunits import ConvBlock, UNetDownx2
from blockunits import DeConvBlockx2



class RedNet(nn.Module):
    """
    self.conv is used to balence dimension
    """
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 zks=3,):
        super(RedNet, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16]]
        # convlution network 
        self.encoder1 = UNetDownx2(nb_channel, kernels[0], is_bn=True)
        _conv1 = ConvBlock(kernels[0], kernels[1], is_bn=True)
        self.conv1 = nn.Sequential(*[_conv1.block[i] for i in range(len(_conv1.block)-1)])
        
        self.encoder2 = UNetDownx2(kernels[0], kernels[1], is_bn=True)
        _conv2 = ConvBlock(kernels[1], kernels[2], is_bn=True)
        self.conv2 = nn.Sequential(*[_conv2.block[i] for i in range(len(_conv2.block)-1)])
        
        self.encoder3 = UNetDownx2(kernels[1], kernels[2], is_bn=True)
        _conv3 = ConvBlock(kernels[2], kernels[3], is_bn=True)
        self.conv3 = nn.Sequential(*[_conv3.block[i] for i in range(len(_conv3.block)-1)])
        
        self.encoder4 = UNetDownx2(kernels[2], kernels[3], is_bn=True)
        _conv4 = ConvBlock(kernels[3], kernels[4], is_bn=True)
        self.conv4 = nn.Sequential(*[_conv4.block[i] for i in range(len(_conv4.block)-1)])
        
        self.encoder5 = UNetDownx2(kernels[3], kernels[4], is_bn=True)

        # deconvlution network
        self.relu = nn.ReLU(True)
        
        _decoder5 = DeConvBlockx2(kernels[4], kernels[4], is_bn=True) 
        self.decoder5 = nn.Sequential(*[_decoder5.block[i] for i in range(len(_decoder5.block)-1)])
        _decoder4 = DeConvBlockx2(kernels[4], kernels[3], is_bn=True) 
        self.decoder4 = nn.Sequential(*[_decoder4.block[i] for i in range(len(_decoder5.block)-1)])
        _decoder3 = DeConvBlockx2(kernels[3], kernels[2], is_bn=True) 
        self.decoder3 = nn.Sequential(*[_decoder3.block[i] for i in range(len(_decoder5.block)-1)])
        _decoder2 = DeConvBlockx2(kernels[2], kernels[1], is_bn=True) 
        self.decoder2 = nn.Sequential(*[_decoder2.block[i] for i in range(len(_decoder5.block)-1)])
        self.decoder1 = DeConvBlockx2(kernels[1], kernels[0], is_bn=True)  
#         generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels[0], nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

    def forward(self, x):
        # encoder
        ex1 = self.encoder1(x)
        _residual1 = ex1
        residual1 = self.conv1(_residual1)
        ex2 = self.encoder2(ex1)
        _residual2 = ex2
        residual2 = self.conv2(_residual2)
        ex3 = self.encoder3(ex2)
        _residual3 = ex3
        residual3 = self.conv3(_residual3)
        ex4 = self.encoder4(ex3)
        _residual4 = ex4
        residual4 = self.conv4(_residual4)
        ex5 = self.encoder5(ex4)
        # decoder 
        _dx5 = self.decoder5(ex5)
        dx5 = self.relu(_dx5 + residual4)
        _dx4 = self.decoder4(dx5)
        dx4 = self.relu(_dx4 + residual3)
        _dx3 = self.decoder3(dx4)
        dx3 = self.relu(_dx3 + residual2)
        _dx2 = self.decoder2(dx3)
        dx2 = self.relu(_dx2 + residual1)
        dx1 = self.decoder1(dx2)

        return self.outconv1(dx1)
        
if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = torch.FloatTensor(
            np.random.random((1, nb_channel, 224, 224)))

    generator = RedNet(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("RedNet->:")
    print(" Network output ", gen_y.shape)

