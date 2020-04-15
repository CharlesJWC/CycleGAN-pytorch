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
''' Discriminator Network (PatchGAN discriminator) '''
class Discriminator(nn.Module):
    ''' Initialization '''
    #===========================================================================
    def __init__(self, input_channel=3):
        super(Discriminator, self).__init__()

        # Discriminator layers
        ## Input layer
        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=64,
                            kernel_size=4, stride=2, padding=1, bias=True)
        # No batch norm
        self.lrelu1 = nn.LeakyReLU(0.2)

        ## Hidden layer1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128,
                            kernel_size=4, stride=2, padding=1, bias=True)
        self.in2 = nn.InstanceNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.2)

        ## Hidden layer2
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256,
                            kernel_size=4, stride=2, padding=1, bias=True)
        self.in3 = nn.InstanceNorm2d(256)
        self.lrelu3 = nn.LeakyReLU(0.2)

        ## Hidden layer3
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512,
                            kernel_size=4, stride=1, padding=1, bias=True)
        self.in4 = nn.InstanceNorm2d(512)
        self.lrelu4 = nn.LeakyReLU(0.2)

        ## Output layer
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=1,
                            kernel_size=4, stride=1, padding=1, bias=True)
        # No batch norm
        # self.sigmoid5 = nn.Sigmoid()

        ## parameters initialization
        self.params_init()



    #===========================================================================
    ''' Parameters initialization '''
    def params_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.normal_(layer.weight.data, 0.0, 0.02)
                nn.init.constant_(layer.bias.data, 0.0)

    #===========================================================================
    ''' Forward from the image x '''
    def forward(self, x):
        # x : Cin x H x W

        # Input layer
        out = self.conv1(x)
        out = self.lrelu1(out)
        # out : 64 x H/2 x H/2

        # Hidden layer1
        out = self.conv2(out)
        out = self.in2(out)
        out = self.lrelu2(out)
        # out : 128 x H/4 x H/4

        # Hidden layer2
        out = self.conv3(out)
        out = self.in3(out)
        out = self.lrelu3(out)
        # out : 256 x H/8 x W/8

        # Hidden layer3
        out = self.conv4(out)
        out = self.in4(out)
        out = self.lrelu4(out)
        # out : 512 x H/8-1 x W/8-1

        # Output layer
        out = self.conv5(out)
        # out : 1 x H/8-2 x W/8-2

        return out

    #===========================================================================
    ''' Forward from the image x to get attantion map '''
    def forward_attantionmap(self, x):
        # x : Cin x H x W

        # Input layer
        out = self.conv1(x)
        out = self.lrelu1(out)
        # out : 64 x H/2 x H/2

        # Hidden layer1
        out = self.conv2(out)
        out = self.in2(out)
        out = self.lrelu2(out)
        # out : 128 x H/4 x H/4

        # Hidden layer2
        out = self.conv3(out)
        out = self.in3(out)
        out = self.lrelu3(out)
        # out : 256 x H/8 x W/8

        # Hidden layer3
        out = self.conv4(out)
        out = self.in4(out)
        attention_tensor = self.lrelu4(out)
        # attention_tensor : 512 x H/8-1 x W/8-1

        # Init attention map
        attention_map = torch.zeros(attention_tensor.size()[2:],
                                    dtype=attention_tensor.dtype,
                                    device=attention_tensor.device)

        # Sum up all the attention channel
        attention_tensor = attention_tensor.squeeze(0)
        for attention_channel in attention_tensor:
            attention_map += attention_channel

        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) \
                        / (attention_map.max() - attention_map.min())

        # Upscale attention map to same size as x
        attention_map = attention_map.unsqueeze(0).unsqueeze(0)
        attention_map = F.interpolate(attention_map, size=x.size()[2:],
                                    mode='bilinear', align_corners=True)
        # attention_map : 1 x H x W

        return attention_map
