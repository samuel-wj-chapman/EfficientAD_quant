#!/usr/bin/python
# -*- coding: utf-8 -*-
from torch import nn
from torchvision.datasets import ImageFolder
import torch


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

'''
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

        return map_combined
    
'''


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

    def forward(self, input_image):
        # Forward pass through each model
        #teacher_output = self.teacher(input_image).detach()
        #student_output = self.student(input_image).detach()
        #autoencoder_output = self.autoencoder(input_image).detach()
        teacher_output = self.teacher(input_image)
        student_output = self.student(input_image)
        autoencoder_output = self.autoencoder(input_image)
        teacher_output = (teacher_output - self.teacher_mean) / self.teacher_std
        # Split student output for comparison with teacher and autoencoder outputs
        student_output_st = student_output[:, :self.out_channels]
        student_output_ae = student_output[:, self.out_channels:]
        mse_st= (teacher_output - student_output_st) * (teacher_output - student_output_st)
        mse_ae= (autoencoder_output - student_output_ae) * (autoencoder_output - student_output_ae)


        
        # Calculate MSE between teacher-student and autoencoder-student
        #mse_st = torch.mean((teacher_output - student_output_st) * (teacher_output - student_output_st), dim=1, keepdim=True)
        #mse_ae = torch.mean((autoencoder_output - student_output_ae) * (autoencoder_output - student_output_ae), dim=1, keepdim=True)
        #if self.q_st_start is not None:
        #    mse_st = 0.1 * (mse_st - self.q_st_start) / (self.q_st_end - self.q_st_start)
        #if self.q_ae_start is not None:
        #    mse_ae = 0.1 * (mse_ae - self.q_ae_start) / (self.q_ae_end - self.q_ae_start)
        ## Combine the MSE maps
        #map_combined = 0.5 * mse_st + 0.5 * mse_ae

        #return teacher_output, student_output, autoencoder_output
        return mse_st, mse_ae