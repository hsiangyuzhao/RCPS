import torch
import torch.nn as nn
import torch.nn.functional as F
from base.base_modules import *


class Backbone(nn.Module):
    """
    Model backbone to extract features
    """
    def __init__(self, input_channels=3, channels=(32, 64, 128, 256, 512), strides=(2, 2, 2, 2),
                 use_dropout=False, **kwargs):
        """
        Args:
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
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


class UNet(nn.Module):
    def __init__(self, num_classes, input_channels=1, channels=(16, 32, 64, 128, 256),
                 strides=(1, 2, 2, 2, 2), use_dropout=False, **kwargs):
        """
        Args:
            num_classes: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=use_dropout, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=use_dropout, **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=use_dropout, **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=use_dropout, **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)

        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0)], dim=1))

        out = dict()
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out


class ProjectorUNet(nn.Module):
    """
    Deep supervised U-Net with a projector layer for contrastive learning
    """
    def __init__(self, num_classes, input_channels=1, channels=(16, 32, 64, 128, 256), project_dim=64,
                 strides=(1, 2, 2, 2, 2), use_dropout=False, **kwargs):
        """
        Args:
            num_classes: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            project_dim: number of channels in the projector layer
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=use_dropout, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 320 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], use_dropout=use_dropout, **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], use_dropout=use_dropout, **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], use_dropout=use_dropout, **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)
        # projector
        self.projector = nn.Sequential(nn.Conv3d(nb_filter[2], nb_filter[2], kernel_size=1),
                                       nn.PReLU(),
                                       nn.Conv3d(nb_filter[2], project_dim, kernel_size=1))

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0)], dim=1))

        out = dict()
        out['project'] = self.projector(x2_2)
        out['project_map'] = F.interpolate(self.convds0(x0_4), size=x2_2.shape[2:], mode='trilinear', align_corners=False)
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out


class DSUNet(nn.Module):
    """
    Deep supervised U-Net
    """
    def __init__(self, num_classes, input_channels=1, channels=(16, 32, 64, 128, 256),
                 strides=(1, 2, 2, 2, 2), use_dropout=False, **kwargs):
        """
        Args:
            num_classes: the number of classes in segmentation
            input_channels: the number of input channels
            channels: length-5 tuple, define the number of channels in each stage
            project_dim: number of channels in the projector layer
            strides: tuple, define the stride in each stage
            use_dropout: bool, whether to use dropout
            **kwargs: other args define activation and normalization
        """
        super().__init__()
        self.backbone = Backbone(input_channels=input_channels, channels=channels, strides=strides,
                                 use_dropout=use_dropout, **kwargs)
        nb_filter = self.backbone.nb_filter

        res_unit = ResBlock if nb_filter[-1] <= 128 else ResBottleneck
        self.conv3_1 = res_unit(nb_filter[3] + nb_filter[4], nb_filter[3], **kwargs)
        self.conv2_2 = res_unit(nb_filter[2] + nb_filter[3], nb_filter[2], **kwargs)
        self.conv1_3 = res_unit(nb_filter[1] + nb_filter[2], nb_filter[1], **kwargs)
        self.conv0_4 = res_unit(nb_filter[0] + nb_filter[1], nb_filter[0], use_dropout=use_dropout, **kwargs)

        # deep supervision
        self.convds3 = nn.Conv3d(nb_filter[3], num_classes, kernel_size=1)
        self.convds2 = nn.Conv3d(nb_filter[2], num_classes, kernel_size=1)
        self.convds1 = nn.Conv3d(nb_filter[1], num_classes, kernel_size=1)
        self.convds0 = nn.Conv3d(nb_filter[0], num_classes, kernel_size=1)

    def upsample(self, inputs, target):
        return F.interpolate(inputs, size=target.shape[2:], mode='trilinear', align_corners=False)

    def forward(self, x):
        size = x.shape[2:]
        x0, x1, x2, x3, x4 = self.backbone(x)

        x3_1 = self.conv3_1(torch.cat([x3, self.upsample(x4, x3)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2, self.upsample(x3_1, x2)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1, self.upsample(x2_2, x1)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0, self.upsample(x1_3, x0)], dim=1))

        out = dict()
        out['level3'] = F.interpolate(self.convds3(x3_1), size=size, mode='trilinear', align_corners=False)
        out['level2'] = F.interpolate(self.convds2(x2_2), size=size, mode='trilinear', align_corners=False)
        out['level1'] = F.interpolate(self.convds1(x1_3), size=size, mode='trilinear', align_corners=False)
        out['out'] = F.interpolate(self.convds0(x0_4), size=size, mode='trilinear', align_corners=False)
        return out

    def get_multi_loss(self, criterion, outputs, target, is_ds=True):
        if is_ds:
            multi_loss = sum([criterion(item, target) for item in outputs.values()])
        else:
            multi_loss = criterion(outputs['out'], target)
        return multi_loss
