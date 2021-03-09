#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:53:41 2018

@author: kaku

DeconvNet:
https://arxiv.org/pdf/1505.04366.pdf
"""

import sys
sys.path.append('./models')
import numpy as np
import torch
import torch.nn as nn
from blockunits import UNetDownx2, UNetDownx3
from blockunits import SegNetUpx2, SegNetUpx3

class DeconvNet(nn.Module):
    """
    encoder is VGG 16-layer backend
    """
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel=64,
                 zks=3,):
        super(DeconvNet, self).__init__()
        kernels = [x * base_kernel for x in [1, 2, 4, 8, 16, 32]]
        # convlution network + pooling: encoder
        self.encoder1 = UNetDownx2(nb_channel, kernels[0], is_bn=True)
        self.maxpool1 = nn.MaxPool2d(2, return_indices=True)

        self.encoder2 = UNetDownx2(kernels[0], kernels[1], is_bn=True)
        self.maxpool2 = nn.MaxPool2d(2, return_indices=True)
        
        self.encoder3 = UNetDownx3(kernels[1], kernels[2], is_bn=True)
        self.maxpool3 = nn.MaxPool2d(2, return_indices=True)
        
        self.encoder4 = UNetDownx3(kernels[2], kernels[3], is_bn=True)
        self.maxpool4 = nn.MaxPool2d(2, return_indices=True)
        
        self.encoder5 = UNetDownx3(kernels[3], kernels[4], is_bn=True)
        self.maxpool5 = nn.MaxPool2d(2, return_indices=True)
        # full connected layer
        
        self.encoderfc = nn.Sequential(
            nn.Conv2d(kernels[4], kernels[5], 4),
            nn.ReLU(inplace=True),)
        
        self.decoderfc = nn.Sequential(
            nn.ConvTranspose2d(kernels[5], kernels[5], 1),
            nn.ReLU(inplace=True),)
        
        self.decoderinit = nn.Sequential(
            nn.ConvTranspose2d(kernels[5], kernels[4], 4),
            nn.BatchNorm2d(kernels[4]),
            nn.ReLU(inplace=True),)
        
        self.decoder5 = SegNetUpx3(
            kernels[4], kernels[3], is_deconv=True, is_bn=True)

        self.decoder4 = SegNetUpx3(
            kernels[3], kernels[2], is_deconv=True, is_bn=True)
        
        self.decoder3 = SegNetUpx3(
            kernels[2], kernels[1], is_deconv=True, is_bn=True)
        
        self.decoder2 = SegNetUpx2(
            kernels[1], kernels[0], is_deconv=True, is_bn=True)

        self.decoder1 = SegNetUpx2(
            kernels[0], nb_class, is_deconv=True, is_bn=True)        

#         generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(nb_class, nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

    def forward(self, x):
        
        ex11 = self.encoder1(x)
#        print(ex11.shape)
        ex12, max_1_indices = self.maxpool1(ex11)
#        print(ex12.shape)
        ex21 = self.encoder2(ex12)
#        print(ex21.shape)
        ex22, max_2_indices = self.maxpool2(ex21)
#        print(ex22.shape)
        ex31 = self.encoder3(ex22)
#        print(ex31.shape)
        ex32, max_3_indices = self.maxpool3(ex31)
#        print(ex31.shape)
        ex41 = self.encoder4(ex32)
#        print(ex41.shape)
        ex42, max_4_indices = self.maxpool4(ex41)
#        print(ex41.shape)
        ex51 = self.encoder5(ex42)
#        print(ex51.shape)
        ex52, max_5_indices = self.maxpool5(ex51)
#        print(ex52.shape)
        
        exfc = self.encoderfc(ex52)
#        print(exfc.shape)
        dxfc = self.decoderfc(exfc)
#        print(dxfc.shape)
        dxinit = self.decoderinit(dxfc)
#        print(dxinit.shape)
        dx5 = self.decoder5(dxinit, max_5_indices, ex51.size())
#        print(dx5.shape)
        dx4 = self.decoder4(dx5, max_4_indices, ex41.size())
#        print(dx4.shape)
        dx3 = self.decoder3(dx4, max_3_indices, ex31.size())
#        print(dx3.shape)
        dx2 = self.decoder2(dx3, max_2_indices, ex21.size())
#        print(dx2.shape)
        dx1 = self.decoder1(dx2, max_1_indices, ex11.size())
#        print(dx1.shape)
        return self.outconv1(dx1)
        
if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    x = torch.FloatTensor(
            np.random.random((1, nb_channel, 224, 224)))

    generator = DeconvNet(nb_channel, nb_class, base_kernel)
    gen_y = generator(x)
    print("DeconvNet->:")
    print(" Network output ", gen_y.shape)

                

            

