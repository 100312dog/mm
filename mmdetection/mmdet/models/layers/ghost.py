import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.layers import SELayer
import math


class GhostModule(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            ratio=2,
            dw_kernel_size=3,
            stride=1,
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU")
    ):
        super().__init__()
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = ConvModule(
            in_channels,
            init_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.cheap_operation = ConvModule(
            init_channels,
            new_channels,
            kernel_size=dw_kernel_size,
            stride=1,
            padding=dw_kernel_size // 2,
            groups=new_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        dw_kernel_size=3,
        stride=1,
        norm_cfg=dict(type="BN"),
        act_cfg=dict(type="ReLU"),
        se_cfg=None
    ):
        super().__init__()
        self.with_se = se_cfg is not None
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(
            in_channels,
            mid_channels,
            kernel_size=1,
            ratio=2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # Depth-wise convolution
        if self.stride > 1:
            self.dw = ConvModule(
                mid_channels,
                mid_channels,
                kernel_size=dw_kernel_size,
                stride=stride,
                padding=dw_kernel_size // 2,
                groups=mid_channels,
                norm_cfg=norm_cfg,
                act_cfg=None)

        # Squeeze-and-excitation
        if self.with_se:
            assert isinstance(se_cfg, dict)
            self.se = SELayer(**se_cfg)

        # Point-wise linear projection
        self.ghost2 = GhostModule(
            mid_channels,
            out_channels,
            kernel_size=1,
            ratio=2,
            norm_cfg=norm_cfg,
            act_cfg=None)

        # shortcut
        if in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = DepthwiseSeparableConvModule(
                in_channels,
                out_channels,
                kernel_size=dw_kernel_size,
                stride=stride,
                padding=dw_kernel_size // 2,
                norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_act_cfg=None,
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.dw(x)

        # Squeeze-and-excitation
        if self.with_se:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


