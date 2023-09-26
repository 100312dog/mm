import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.layers import GhostBottleneck
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.registry import MODELS
from mmengine.model import BaseModule


class GhostBlocks(nn.Module):
    """Stack of GhostBottleneck used in GhostPAN.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        activation (str): Name of activation function. Default: LeakyReLU.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=5,
        expand=1,
        num_blocks=1,
        use_res=False,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
    ):
        super(GhostBlocks, self).__init__()
        self.use_res = use_res
        if use_res:
            self.reduce_conv = ConvModule(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=None,
                act_cfg=act_cfg
            )
        blocks = []
        for _ in range(num_blocks):
            blocks.append(
                GhostBottleneck(
                    in_channels,
                    int(out_channels * expand),
                    out_channels,
                    dw_kernel_size=kernel_size,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.blocks(x)
        if self.use_res:
            out = out + self.reduce_conv(x)
        return out

@MODELS.register_module()
class GhostPAN(BaseModule):
    """Path Aggregation Network with Ghost block.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 3
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        kernel_size (int): Kernel size of depthwise convolution. Default: 5.
        expand (int): Expand ratio of GhostBottleneck. Default: 1.
        num_blocks (int): Number of GhostBottlecneck blocks. Default: 1.
        use_res (bool): Whether to use residual connection. Default: False.
        num_extra_level (int): Number of extra conv layers for more feature levels.
            Default: 0.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(scale_factor=2, mode='nearest')`
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN')
        activation (str): Activation layer name.
            Default: LeakyReLU.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            kernel_size=5,
            use_depthwise=True,
            ghost_block_cfg=dict(expand=1, num_blocks=1, use_res=False),
            upsample_cfg=dict(scale_factor=2, mode="bilinear"),
            norm_cfg=dict(type="BN"),
            act_cfg=dict(type="ReLU"),
            init_cfg=None):
        super().__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.kernel_size = kernel_size
        self.upsample_cfg = upsample_cfg

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level

        conv_func = DepthwiseSeparableConvModule if use_depthwise else ConvModule

        # build laterals
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)

        # build top down pathway
        self.fpn_convs = nn.ModuleList()
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            fpn_conv = GhostBlocks(
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **ghost_block_cfg
            )
            self.fpn_convs.append(fpn_conv)

        # build bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level - 1):
            d_conv = conv_func(
                out_channels,
                out_channels,
                self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            pafpn_conv = GhostBlocks(
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                **ghost_block_cfg
            )
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

        # add extra conv layers
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        assert extra_levels <= 1
        if extra_levels == 1:
            self.first_top_conv = conv_func(
                out_channels,
                out_channels,
                self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.second_top_conv = conv_func(
                out_channels,
                out_channels,
                self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        inter_outs = [laterals[-1]]
        for i in range(used_backbone_levels - 1, 0, -1):
            feat_low = laterals[i - 1]
            feat_high = inter_outs[0]
            feat_upsample = F.interpolate(feat_high, **self.upsample_cfg)
            feat_fused = self.fpn_convs[used_backbone_levels - 1 - i](torch.cat((feat_upsample, feat_low), dim=1))
            inter_outs.insert(0, feat_fused)

        # part 2: add bottom-up path
        outs = [inter_outs[0]]
        for i in range(0, used_backbone_levels - 1):
            feat_low = outs[-1]
            feat_high = inter_outs[i + 1]
            feat_downsample = self.downsample_convs[i](feat_low)
            feat_fused = self.pafpn_convs[i](torch.cat((feat_downsample, feat_high), dim=1))
            outs.append(feat_fused)

        # part 3: add extra levels
        if self.num_outs > len(outs):
            top_features = self.first_top_conv(laterals[-1])
            top_features = top_features + self.second_top_conv(outs[-1])
            outs.append(top_features)

        return tuple(outs)