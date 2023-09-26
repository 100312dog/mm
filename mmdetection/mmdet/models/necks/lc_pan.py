# Copyright (c) 2022 Deepblue Authors. All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from mmdet.models.layers import DepthwiseSeparableConvSEModule
from mmengine.model import BaseModule
from mmdet.registry import MODELS
from mmengine.model import kaiming_init

@MODELS.register_module()
class LCPAN(BaseModule):
    """Path Aggregation Network with LCNet module.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        kernel_size (int): The conv2d kernel size of this Module.
        num_features (int): Number of output features of CSPPAN module.
        num_csp_blocks (int): Number of bottlenecks in CSPLayer. Default: 1
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: True
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 kernel_size=5,
                 use_depthwise=True,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(scale_factor=2, mode="nearest"),
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.kernel_size = kernel_size
        self.no_norm_on_lateral = no_norm_on_lateral
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
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            self.lateral_convs.append(l_conv)

        NET_CONFIG = {
            # k, in_c, out_c, stride, use_se
            "block1": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ],
            "block2": [
                [kernel_size, out_channels * 2, out_channels * 2, 1, False],
                [kernel_size, out_channels * 2, out_channels, 1, False],
            ]
        }

        # build top down pathway
        self.fpn_convs = nn.ModuleList()
        for i in range(self.backbone_end_level - 1, self.start_level, -1):
            fpn_conv = nn.Sequential(
                *[DepthwiseSeparableConvSEModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,

                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    se_cfg=dict(channels=in_c,
                                ratio=4,
                                act_cfg=(dict(type='ReLU'), dict(type='HSigmoid'))) if se else None)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["block1"])]
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
            pafpn_conv = nn.Sequential(
                *[DepthwiseSeparableConvSEModule(
                    in_channels=in_c,
                    out_channels=out_c,
                    kernel_size=k,
                    stride=s,
                    padding=k // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    se_cfg=dict(channels=in_c,
                                ratio=4,
                                act_cfg=(dict(type='ReLU'), dict(type='HSigmoid'))) if se else None)
                    for i, (k, in_c, out_c, s, se) in enumerate(NET_CONFIG["block2"])]
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

    def init_weights(self) -> None:
        kaiming_init(self.first_top_conv.depthwise_conv.conv, distribution="uniform")
        kaiming_init(self.first_top_conv.pointwise_conv.conv, distribution="uniform")
        kaiming_init(self.second_top_conv.depthwise_conv.conv, distribution="uniform")
        kaiming_init(self.second_top_conv.pointwise_conv.conv, distribution="uniform")
        for layer in self.downsample_convs:
            kaiming_init(layer.depthwise_conv.conv, distribution="uniform")
            kaiming_init(layer.pointwise_conv.conv, distribution="uniform")
