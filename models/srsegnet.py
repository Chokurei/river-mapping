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
from blockunits import *
import torch.nn.init as init


class SRSegNet(nn.Module):
    def __init__(self,
                 nb_channel=3,
                 nb_class=1,
                 base_kernel_sr=64,
                 base_kernel_seg=64,
                 upscale_factor=4):
        super(SRSegNet, self).__init__()

        # SR
        kernels_sr = [int(x * base_kernel_sr) for x in [1, 1, 1 / 2]]
        self.relu = nn.ReLU(True)
        self.conv1_sr = nn.Conv2d(
            nb_channel, kernels_sr[0], kernel_size=5, stride=1, padding=2)
        self.conv2_sr = nn.Conv2d(
            kernels_sr[0], kernels_sr[1], kernel_size=3, stride=1, padding=1)
        self.conv3_sr = nn.Conv2d(
            kernels_sr[1], kernels_sr[2], kernel_size=3, stride=1, padding=1)
        self.conv4_sr = nn.Conv2d(
            kernels_sr[2], nb_channel * upscale_factor ** 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        # segmentation
        kernels_seg = [x * base_kernel_seg for x in [1, 2, 4, 8, 16]]
        # down&pooling
        self.downblock1 = UNetDownx2(
            nb_channel, kernels_seg[0])
        self.maxpool1 = nn.MaxPool2d(2)

        self.downblock2 = UNetDownx2(
            kernels_seg[0], kernels_seg[1])
        self.maxpool2 = nn.MaxPool2d(2)

        self.downblock3 = UNetDownx2(
            kernels_seg[1], kernels_seg[2])
        self.maxpool3 = nn.MaxPool2d(2)

        self.downblock4 = UNetDownx2(
            kernels_seg[2], kernels_seg[3])
        self.maxpool4 = nn.MaxPool2d(2)

        # center convolution
        self.center = ConvBlock(kernels_seg[3], kernels_seg[4])

        # up&concating
        self.upblock4 = UNetUpx2(
            kernels_seg[4], kernels_seg[3])

        self.upblock3 = UNetUpx2(
            kernels_seg[3], kernels_seg[2])

        self.upblock2 = UNetUpx2(
            kernels_seg[2], kernels_seg[1])

        self.upblock1 = UNetUpx2(
            kernels_seg[1], kernels_seg[0])

        # generate output
        self.outconv1 = nn.Sequential(
            nn.Conv2d(kernels_seg[0], nb_class, 1),
            nn.Sigmoid() if nb_class==1 else nn.Softmax(dim=1),)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1_sr(x))
        x = self.relu(self.conv2_sr(x))
        x = self.relu(self.conv3_sr(x))
        x = self.pixel_shuffle(self.conv4_sr(x))

        dx11 = self.downblock1(x)
        dx12 = self.maxpool1(dx11)

        dx21 = self.downblock2(dx12)
        dx22 = self.maxpool2(dx21)

        dx31 = self.downblock3(dx22)
        dx32 = self.maxpool3(dx31)

        dx41 = self.downblock4(dx32)
        dx42 = self.maxpool4(dx41)

        cx = self.center(dx42)

        ux4 = self.upblock4(cx, dx41)
        ux3 = self.upblock3(ux4, dx31)
        ux2 = self.upblock2(ux3, dx21)
        ux1 = self.upblock1(ux2, dx11)

        return x, self.outconv1(ux1)

    def _initialize_weights(self):
        init.orthogonal_(self.conv1_sr.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2_sr.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3_sr.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4_sr.weight, init.calculate_gain('relu'))

if __name__ == "__main__":
    # Hyper Parameters
    nb_channel = 3
    nb_class = 1
    base_kernel = 24
    # x = torch.tensor(np.random.random((1, nb_channel, 224, 224)), dtype=torch.float32)
    x = torch.tensor(np.random.random((1, nb_channel, 112, 112)), dtype=torch.float32)
    generator = SRSegNet(nb_channel, nb_class, base_kernel_sr=24, base_kernel_seg=24, upscale_factor=2)
    _sr, gen_y = generator(x)
    print("SRSegNet->:")
    print(" Network output ", _sr.shape, gen_y.shape)

