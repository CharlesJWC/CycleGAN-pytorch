#-*- coding:utf-8 -*-
'''
[AI502] Deep Learning Assignment
"Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" Implementation
20193640 Jungwon Choi
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#===============================================================================
''' Generator Network (6-block Resnet generator) '''
class Generator(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, input_channel=3, output_channel=3, num_resblock=9):
        super(Generator, self).__init__()

        # Generator layers
        ## Input layer
        self.reflect1 = nn.ReflectionPad2d(padding=3)
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=64,
                            kernel_size=7, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(num_features=64)
        self.relu1 = nn.ReLU()

        ## Downsampling layer1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=3, stride=2, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(128)
        self.relu2 = nn.ReLU()
        ## Downsampling layer2
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=3, stride=2, padding=1, bias=True)
        self.in3 = nn.InstanceNorm2d(256)
        self.relu3 = nn.ReLU()

        ## Residual blocks
        resblocks= [ResnetBlock(256)]*num_resblock
        self.resblocks = nn.Sequential(*resblocks)

        ## Upsampling layer1
        self.deconv4 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                            kernel_size=3, stride=2, padding=1,
                            output_padding=1, bias=True)
        self.in4 = nn.InstanceNorm2d(128)
        self.relu4 = nn.ReLU()

        ## Upsampling layer2
        self.deconv5 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                            kernel_size=3, stride=2, padding=1,
                            output_padding=1, bias=True)
        self.in5 = nn.InstanceNorm2d(64)
        self.relu5 = nn.ReLU()

        ## Output layer
        self.reflect6 = nn.ReflectionPad2d(padding=3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=output_channel,
                            kernel_size=7, stride=1, padding=0, bias=True)
        # No batch norm
        self.tanh6 = nn.Tanh()

        ## parameters initialization
        self.params_init()

    #===========================================================================
    ''' Parameters initialization '''
    def params_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)
            if isinstance(layer, nn.ConvTranspose2d):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)

    #===========================================================================
    ''' Forward from the image x '''
    def forward(self, x):
        # x : Cin x H x W

        # Input layer
        out = self.reflect1(x)
        # out : Cin x H+6 x W+6
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu1(out)
        # out : 64 x H x W

        # Downsampling layer1
        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu2(out)
        # out : 128 x H/2 x W/2

        # Downsampling layer2
        out = self.conv3(out)
        out = self.in3(out)
        out = self.relu3(out)
        # out : 256 x H/4 x W/4

        # Resnet blocks
        out = self.resblocks(out)
        # out : 256 x H/4 x W/4

        # Upsampling layer1
        out = self.deconv4(out)
        out = self.in4(out)
        out = self.relu4(out)
        # out : 128 x H/2 x W/2

        # Upsampling layer2
        out = self.deconv5(out)
        out = self.in5(out)
        out = self.relu5(out)
        # out : 64 x H x W

        # Output layer
        out = self.reflect6(out)
        # out : 64 x H+6 x W+6
        out = self.conv6(out)
        out = self.tanh6(out)
        # out : Cout x H x W

        return out

    #===========================================================================
    ''' Forward from the image x and get feature map'''
    def forward_featuremap(self, x):
        # x : Cin x H x W

        # Input layer
        out = self.reflect1(x)
        # out : Cin x H+6 x W+6
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu1(out)
        # out : 64 x H x W

        # Downsampling layer1
        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu2(out)
        # out : 128 x H/2 x W/2

        # Downsampling layer2
        out = self.conv3(out)
        out = self.in3(out)
        out = self.relu3(out)
        # out : 256 x H/4 x W/4

        # Resnet blocks
        out = self.resblocks(out)
        # out : 256 x H/4 x W/4

        # Upsampling layer1
        out = self.deconv4(out)
        out = self.in4(out)
        feature_map = self.relu4(out)
        # feature_map : 128 x H/2 x W/2

        return feature_map


#===============================================================================
''' Resnet Block '''
class ResnetBlock(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, input_channel):
        super(ResnetBlock, self).__init__()

        # Block layer1
        self.reflect1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(input_channel, input_channel,
                            kernel_size=3, stride=1, padding=0, bias=True)
        self.in1 =  nn.InstanceNorm2d(num_features=input_channel)
        self.relu1 = nn.ReLU()

        # Block layer2
        self.reflect2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(input_channel, input_channel,
                            kernel_size=3, stride=1, padding=0, bias=True)
        self.in2 =  nn.InstanceNorm2d(num_features=input_channel)

    #===========================================================================
    ''' Forward Resnet Block '''
    def forward(self, x):
        # x : C x H x W

        # Block layer1
        out = self.reflect1(x)
        # out : C x H+2 x W+2
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu1(out)
        # out : C x H x W

        # Block layer2
        out = self.reflect2(out)
        # out : C x H+2 x W+2
        out = self.conv2(out)
        out = self.in2(out)
        # out : C x H x W

        # Add skip connections
        out = x + out
        # out : C x H x W

        return out
