#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch
import torch.nn.functional as F


def get_autoencoder(out_channels=384):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=8),
        # decoder
        nn.Upsample(size=3, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=8, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=15, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=32, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=63, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=127, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1,
                  padding=2),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(size=56, mode='bilinear'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                  padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3,
                  stride=1, padding=1)
    )

import torch.nn as nn

import torch
import torch.nn as nn

class AutoencoderImproved(nn.Module):
    def __init__(self, out_channels=384):
        super(AutoencoderImproved, self).__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 256 -> 128
            nn.ReLU(inplace=True)
        )  # Output: (N, 32, 128, 128)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.ReLU(inplace=True)
        )  # Output: (N, 32, 64, 64)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.ReLU(inplace=True)
        )  # Output: (N, 64, 32, 32)
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(inplace=True)
        )  # Output: (N, 64, 16, 16)
        
        self.enc5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(inplace=True)
        )  # Output: (N, 64, 8, 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(inplace=True)
        )  # Output: (N, 64, 4, 4)
        
        # Decoder
        self.up5 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 4 -> 8
        self.dec5 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 8 -> 16
        self.dec4 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.up3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  # 16 -> 32
        self.dec3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 32 -> 64
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.up1 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1)  # 64 -> 128
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.up0 = nn.ConvTranspose2d(32, 64, kernel_size=4, stride=2, padding=1)  # 128 -> 256
        self.dec0 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        # Final resizing to 56x56
        self.output_resize = nn.AdaptiveAvgPool2d((56, 56))

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # (N, 32, 128, 128)
        e2 = self.enc2(e1)   # (N, 32, 64, 64)
        e3 = self.enc3(e2)   # (N, 64, 32, 32)
        e4 = self.enc4(e3)   # (N, 64, 16, 16)
        e5 = self.enc5(e4)   # (N, 64, 8, 8)
        b = self.bottleneck(e5)  # (N, 64, 4, 4)
        
        # Decoder with additive skip connections
        d5 = self.up5(b)         # (N, 64, 4, 4) -> (N, 64, 8, 8)
        d5 = self.dec5(d5 + e5)  # Add skip connection from e5
        
        d4 = self.up4(d5)        # (N, 64, 8, 8) -> (N, 64, 16, 16)
        d4 = self.dec4(d4 + e4)  # Add skip connection from e4
        
        d3 = self.up3(d4)        # (N, 64, 16, 16) -> (N, 64, 32, 32)
        d3 = self.dec3(d3 + e3)  # Add skip connection from e3
        
        d2 = self.up2(d3)        # (N, 64, 32, 32) -> (N, 32, 64, 64)
        d2 = self.dec2(d2 + e2)  # Add skip connection from e2
        
        d1 = self.up1(d2)        # (N, 32, 64, 64) -> (N, 32, 128, 128)
        d1 = self.dec1(d1 + e1)  # Add skip connection from e1
        
        d0 = self.up0(d1)        # (N, 32, 128, 128) -> (N, 64, 256, 256)
        d0 = self.dec0(d0)       # Output: (N, out_channels, 256, 256)
        
        # Resize output to 56x56
        out = self.output_resize(d0)
        return out


def get_autoencoder_tiny(out_channels=192):  # Adjusted for output size 58x58
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),  # 128x128
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 64x64
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 32x32
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),  # 16x16
        nn.ReLU(inplace=True),
        # decoder
        nn.Upsample(scale_factor=2, mode='bilinear'),  # 32x32
        nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),  # 32x32
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Upsample(scale_factor=2, mode='bilinear'),  # 64x64
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),  # 64x64
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0),  # 62x62
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=5, stride=1, padding=0)  # 58x58
    )

def get_pdn_small(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=256, out_channels=out_channels, kernel_size=4)
    )


def get_pdn_small_tiny(out_channels=192, padding=False):  # Adjusted channel sizes for slight performance increase
    pad_mult = 1 if padding else 0
    # Increase channels by about 10%
    #first_layer_channels = int(64 * 1.2)  # Originally 64
    #second_layer_channels = int(128 * 1.2)  # Originally 128
    first_layer_channels = 96 # to use memory effectively
    second_layer_channels = 160 # to use memory effectively
    print(first_layer_channels, second_layer_channels)
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=first_layer_channels, kernel_size=3, padding=2 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=first_layer_channels, out_channels=second_layer_channels, kernel_size=3, padding=2 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=second_layer_channels, out_channels=second_layer_channels, kernel_size=3, padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=second_layer_channels, out_channels=out_channels, kernel_size=3)
    )


import torch.nn as nn

def get_pdn_small_tinyer(out_channels=192, padding=False):
    pad_mult = 1 if padding else 0
    first_layer_channels = 96  # To use memory effectively
    second_layer_channels = 160  # To use memory effectively
    print(first_layer_channels, second_layer_channels)

    return nn.Sequential(
        # First Convolutional Block
        nn.Conv2d(in_channels=3, out_channels=first_layer_channels, kernel_size=3, padding=2 * pad_mult),
        nn.BatchNorm2d(first_layer_channels),  # Added BatchNorm
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),  # Replaced AvgPool with MaxPool

        # Second Convolutional Block
        nn.Conv2d(in_channels=first_layer_channels, out_channels=second_layer_channels, kernel_size=3, padding=2 * pad_mult),
        nn.BatchNorm2d(second_layer_channels),  # Added BatchNorm
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),  # Replaced AvgPool with MaxPool

        # Third Convolutional Block
        nn.Conv2d(in_channels=second_layer_channels, out_channels=second_layer_channels, kernel_size=3, padding=1 * pad_mult),
        nn.BatchNorm2d(second_layer_channels),  # Added BatchNorm
        nn.ReLU(inplace=True),

        # Output Layer
        nn.Conv2d(in_channels=second_layer_channels, out_channels=out_channels, kernel_size=3)
    )


def get_pdn_medium(out_channels=384, padding=False):
    pad_mult = 1 if padding else 0
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                  padding=3 * pad_mult),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                  padding=1 * pad_mult),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=4),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                  kernel_size=1)
    )

class ImageFolderWithoutTarget(ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        return sample

class ImageFolderWithPath(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = super().__getitem__(index)
        return sample, target, path

def InfiniteDataloader(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


class UnifiedAnomalyDetectionModel(nn.Module):
    def __init__(self, teacher, student, autoencoder, out_channels, teacher_mean, teacher_std, q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
        super(UnifiedAnomalyDetectionModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.autoencoder = autoencoder
        self.out_channels = out_channels
        self.teacher_mean = teacher_mean
        self.teacher_std = teacher_std
        self.q_st_start = q_st_start
        self.q_st_end = q_st_end
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end

    @torch.no_grad()
    def forward(self, input_image):
        # Forward pass through each model
        teacher_output = self.teacher(input_image).detach()
        student_output = self.student(input_image).detach()
        autoencoder_output = self.autoencoder(input_image).detach()
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        # Split student output for comparison with teacher and autoencoder outputs
        student_output_st = student_output[:, :self.out_channels]
        student_output_ae = student_output[:, self.out_channels:]

        # Calculate MSE between teacher-student and autoencoder-student
        mse_st = torch.mean((teacher_output - student_output_st) * (teacher_output - student_output_st), dim=1, keepdim=True)
        mse_ae = torch.mean((autoencoder_output - student_output_ae) * (autoencoder_output - student_output_ae), dim=1, keepdim=True)
        if self.q_st_start is not None:
            mse_st = 0.1 * (mse_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        if self.q_ae_start is not None:
            mse_ae = 0.1 * (mse_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        # Combine the MSE maps
        map_combined = 0.5 * mse_st + 0.5 * mse_ae
        max_value = F.max_pool2d(map_combined, kernel_size=(56, 56))
        #max_value = max_value.squeeze() 
        return  max_value




class UnifiedAnomalyDetectionModel(nn.Module):
    def __init__(self, student, autoencoder, out_channels,q_ae_start=None, q_ae_end=None):
        super(UnifiedAnomalyDetectionModel, self).__init__()

        self.student = student
        self.autoencoder = autoencoder
        self.out_channels = out_channels
        self.q_ae_start = q_ae_start
        self.q_ae_end = q_ae_end

    @torch.no_grad()
    def forward(self, input_image):
        # Forward pass through each model
        student_output = self.student(input_image).detach()
        autoencoder_output = self.autoencoder(input_image).detach()
        # Split student output for comparison with teacher and autoencoder outputs

        student_output_ae = student_output

        # Calculate MSE between teacher-student and autoencoder-student
        mse_ae = torch.mean((autoencoder_output - student_output_ae) * (autoencoder_output - student_output_ae), dim=1, keepdim=True)
        if self.q_ae_start is not None:
            mse_ae = 0.1 * (mse_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        
        max_value = F.max_pool2d(mse_ae, kernel_size=(56, 56))
       
        return  max_value

        import torch
import torch.nn as nn
from torchvision import models

import torch
import torch.nn as nn
from torchvision import models

def get_mobilenet_v2_feature_extractor(out_channels=384):
    # Load MobileNetV2
    mobilenet_v2 = models.mobilenet_v2(pretrained=False)
    
    # Modify the initial convolutional layer to have stride=1
    mobilenet_v2.features[0][0] = nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    
    # Reconfigure the inverted residual blocks
    inverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],  # No downsampling here
        [6, 24, 2, 2],  # Downsampling here
        [6, 32, 3, 2],  # Downsampling here
        [6, 64, 4, 1],  # No further downsampling
    ]

    features = [mobilenet_v2.features[0]]  # Initial layer
    input_channel = 32
    for t, c, n, s in inverted_residual_setting:
        output_channel = c
        for i in range(n):
            stride = s if i == 0 else 1
            features.append(
                models.mobilenet.InvertedResidual(input_channel, output_channel, stride, expand_ratio=t)
            )
            input_channel = output_channel

    # Replace the features
    mobilenet_v2.features = nn.Sequential(*features)
    
    # Remove the classifier
    mobilenet_v2.classifier = nn.Identity()
    
    # Add custom layers to adjust channels and spatial dimensions
    mobilenet_v2.features.add_module('custom_conv', nn.Conv2d(
        in_channels=input_channel,
        out_channels=out_channels,
        
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False
    ))
    nn.init.kaiming_normal_(mobilenet_v2.features.custom_conv.weight, mode='fan_out', nonlinearity='relu')
    
    mobilenet_v2.features.add_module('adjust_conv', nn.Conv2d(
        in_channels=out_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=1,
        padding=0,
        bias=False
    ))
    nn.init.kaiming_normal_(mobilenet_v2.features.adjust_conv.weight, mode='fan_out', nonlinearity='relu')
    
    mobilenet_v2.features.add_module('relu', nn.ReLU(inplace=True))
    
    return mobilenet_v2


