import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_modules import *


__all__ = ['AttentionGate', 'ParallelDecoder', 'FCNHead', 'Backbone', 'UNetDecoder']


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, **kwargs):
        super().__init__()
        self.W_g = ConvNorm(F_g, F_int, kernel_size=1, stride=1, activation=False, **kwargs)

        self.W_x = ConvNorm(F_l, F_int, kernel_size=1, stride=2, activation=False, **kwargs)

        self.psi = nn.Sequential(
            ConvNorm(F_int, 1, kernel_size=1, stride=1, activation=False, **kwargs),
            nn.Sigmoid()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * self.upsample(psi)


class ParallelDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2
        self.conv3_0 = ConvNorm(in_channels[0], self.midchannels, 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], self.midchannels, 1, 1, **kwargs)
        self.conv5_0 = ConvNorm(in_channels[2], self.midchannels, 1, 1, **kwargs)

        self.conv4_5 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)
        self.conv3_4 = ConvNorm(2 * self.midchannels, self.midchannels, 3, **kwargs)

        self.conv_out = nn.Conv3d(3 * self.midchannels, out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        size = x3.shape[2:]

        # first interpolate three feature maps to the same resolution
        f3 = self.conv3_0(x3)  # (None, midchannels, h3, w3)
        f4 = self.conv4_0(F.interpolate(x4, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)
        level5 = self.conv5_0(F.interpolate(x5, size, mode='trilinear', align_corners=False))  # (None, midchannels, h3, w3)

        # fuse feature maps
        level4 = self.conv4_5(torch.cat([f4, level5], dim=1))  # (None, midchannels, h3, w3)
        level3 = self.conv3_4(torch.cat([f3, level4], dim=1))  # (None, midchannels, h3, w3)

        fused_out_reduced = torch.cat([level3, level4, level5], dim=1)

        out = self.conv_out(fused_out_reduced)

        return out


class FCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        assert isinstance(in_channels, (tuple, list)) and len(in_channels) == 3
        self.midchannels = in_channels[0] // 2

        self.conv5_4 = ConvNorm(in_channels[2], in_channels[1], 1, 1, **kwargs)
        self.conv4_0 = ConvNorm(in_channels[1], in_channels[1], 3, 1, **kwargs)
        self.conv4_3 = ConvNorm(in_channels[1], in_channels[0], 1, 1, **kwargs)
        self.conv3_0 = ConvNorm(in_channels[0], in_channels[0], 3, 1, **kwargs)

        self.conv_out = nn.Conv3d(in_channels[0], out_channels, kernel_size=1)

    def forward(self, x3, x4, x5):
        # x1 has the fewest channels and largest resolution
        # x3 has the most channels and the smallest resolution
        x5_up = self.conv5_4(F.interpolate(x5, size=x4.shape[2:], mode='trilinear', align_corners=False))
        x4_refine = self.conv4_0(x5_up + x4)
        x4_up = self.conv4_3(F.interpolate(x4_refine, size=x3.shape[2:], mode='trilinear', align_corners=False))
        x3_refine = self.conv3_0(x4_up + x3)

        out = self.conv_out(x3_refine)

        return out


class Backbone(nn.Module):
    """
    model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                 use_dropout=False, **kwargs):
        super().__init__()
        self.nb_filter = channels
        self.strides = strides + (5 - len(strides)) * (1,)
        res_unit = ResBlock if channels[-1] <= 320 else ResBottleneck

        if kwargs['norm'] == 'GROUP':
            self.conv0_0 = nn.Sequential(
                nn.Conv3d(input_channels, self.nb_filter[0], kernel_size=3, stride=self.strides[0], padding=1),
                nn.ReLU()
            )
        else:
            self.conv0_0 = ResBlock(input_channels, self.nb_filter[0], self.strides[0], **kwargs)
        self.conv1_0 = res_unit(self.nb_filter[0], self.nb_filter[1], self.strides[1], **kwargs)
        self.conv2_0 = res_unit(self.nb_filter[1], self.nb_filter[2], self.strides[2], **kwargs)
        self.conv3_0 = res_unit(self.nb_filter[2], self.nb_filter[3], self.strides[3], **kwargs)
        self.conv4_0 = res_unit(self.nb_filter[3], self.nb_filter[4], self.strides[4], use_dropout=use_dropout, **kwargs)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(x0_0)
        x2_0 = self.conv2_0(x1_0)
        x3_0 = self.conv3_0(x2_0)
        x4_0 = self.conv4_0(x3_0)
        return x0_0, x1_0, x2_0, x3_0, x4_0


class UNetDecoder(nn.Module):
    def __init__(self, nb_filter, **kwargs):
        super().__init__()
        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], **kwargs)

    def upsample(self, inputs, size):
        return F.interpolate(inputs, size=size, mode='trilinear', align_corners=False)

    def forward(self, x0, x1, x2, x3, x4):
        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3.shape[2:])], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2.shape[2:])], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1.shape[2:])], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0.shape[2:])], dim=1))
        return x3_1, x2_2, x1_3, x0_4