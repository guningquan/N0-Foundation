# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPoolingBlock(nn.Module):
    def __init__(self, in_ch: int, bin_sizes=(1, 2, 3, 6)):
        super().__init__()
        self.bin_sizes = bin_sizes
        self.proj_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, 512, kernel_size=1) for _ in bin_sizes]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2], x.shape[3]
        pooled_outputs = [x]
        for conv, bin_size in zip(self.proj_convs, self.bin_sizes):
            kernel_size = (h // bin_size, w // bin_size)
            stride = kernel_size
            pooled = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)
            pooled = F.relu(conv(pooled), inplace=True)
            upsampled = F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=False)
            pooled_outputs.append(upsampled)

        return torch.cat(pooled_outputs, dim=1)


class MultiTaskPSPNet(nn.Module):
    def __init__(self, in_channels: int = 2, num_classes: int = 2, bin_sizes=(1, 2, 3, 6)):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.ppm = PyramidPoolingBlock(256, bin_sizes=bin_sizes)
        psp_out_channels = 256 + len(bin_sizes) * 512

        self.seg_conv1 = nn.Conv2d(psp_out_channels, 256, kernel_size=3, padding=1)
        self.seg_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.seg_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.seg_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(32, num_classes, kernel_size=1)

        self.reg_conv1 = nn.Conv2d(psp_out_channels, 256, kernel_size=3, padding=1)
        self.reg_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.reg_conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.reg_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pressure_head = nn.Conv2d(32, 1, kernel_size=1)
        self.shear_head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        x = F.relu(self.conv1(x), inplace=True)
        x = self.pool1(x)

        x = F.relu(self.conv2(x), inplace=True)
        x = self.pool2(x)

        x = F.relu(self.conv3(x), inplace=True)
        x = self.pool3(x)

        x = F.relu(self.conv4(x), inplace=True)
        x = self.pool4(x)

        x = self.ppm(x)

        seg = F.relu(self.seg_conv1(x), inplace=True)
        seg = F.interpolate(seg, scale_factor=2, mode='bilinear', align_corners=False)
        seg = F.relu(self.seg_conv2(seg), inplace=True)
        seg = F.interpolate(seg, scale_factor=2, mode='bilinear', align_corners=False)
        seg = F.relu(self.seg_conv3(seg), inplace=True)
        seg = F.interpolate(seg, scale_factor=2, mode='bilinear', align_corners=False)
        seg = F.relu(self.seg_conv4(seg), inplace=True)
        seg = F.interpolate(seg, scale_factor=2, mode='bilinear', align_corners=False)
        seg_output = torch.sigmoid(self.seg_head(seg))

        reg = F.relu(self.reg_conv1(x), inplace=True)
        reg = F.interpolate(reg, scale_factor=2, mode='bilinear', align_corners=False)
        reg = F.relu(self.reg_conv2(reg), inplace=True)
        reg = F.interpolate(reg, scale_factor=2, mode='bilinear', align_corners=False)
        reg = F.relu(self.reg_conv3(reg), inplace=True)
        reg = F.interpolate(reg, scale_factor=2, mode='bilinear', align_corners=False)
        reg = F.relu(self.reg_conv4(reg), inplace=True)
        reg = F.interpolate(reg, scale_factor=2, mode='bilinear', align_corners=False)

        pressure_output = self.pressure_head(reg)
        shear_output = self.shear_head(reg)

        if seg_output.shape[2:] != input_size:
            seg_output = F.interpolate(
                seg_output, size=input_size, mode='bilinear', align_corners=False)
            pressure_output = F.interpolate(
                pressure_output, size=input_size, mode='bilinear', align_corners=False)
            shear_output = F.interpolate(
                shear_output, size=input_size, mode='bilinear', align_corners=False)

        return seg_output, pressure_output, shear_output
