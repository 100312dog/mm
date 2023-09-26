# copyright (c) 2021 Deepblue Authors. All Rights Reserve.
import warnings

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet.models.layers import DepthwiseSeparableConvSEModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..utils import make_divisible
from mmengine.model import bias_init_with_prob, constant_init, normal_init, kaiming_init


@MODELS.register_module()
class LCNet(BaseModule):

    NET_CONFIG = {
        "blocks2":
        # k, in_c, out_c, s, use_se
            [[3, 16, 32, 1, False], ],
        "blocks3": [
            [3, 32, 64, 2, False],
            [3, 64, 64, 1, False],
        ],
        "blocks4": [
            [3, 64, 128, 2, False],
            [3, 128, 128, 1, False],
        ],
        "blocks5": [
            [3, 128, 256, 2, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
            [5, 256, 256, 1, False],
        ],
        "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True]]
    }
    
    def __init__(self, 
                 scale=1.0, 
                 feature_maps=[3, 4, 5], 
                 # conv_cfg=None -> dict(type='Conv2d')
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='HSwish'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.scale = scale
        self.feature_maps = feature_maps
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=make_divisible(16 * scale, 8),
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        for block_name, block_configs in self.NET_CONFIG.items():
            self.add_module(block_name,
                            nn.Sequential(
                                *[DepthwiseSeparableConvSEModule(
                                    in_channels=make_divisible(in_c * scale, 8),
                                    out_channels=make_divisible(out_c * scale, 8),
                                    kernel_size=k,
                                    stride=s,
                                    padding=k//2,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=self.act_cfg,
                                    se_cfg=dict(channels=make_divisible(in_c * scale, 8),
                                                ratio=4,
                                                act_cfg=(dict(type='ReLU'), dict(type='HSigmoid'))) if se else None
                                )
                                    for k, in_c, out_c, s, se in block_configs])
                            )

    def forward(self, x):
        outs = []

        x = self.conv1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        outs.append(x)
        x = self.blocks4(x)
        outs.append(x)
        x = self.blocks5(x)
        outs.append(x)
        x = self.blocks6(x)
        outs.append(x)
        outs = [o for i, o in enumerate(outs) if i + 2 in self.feature_maps]
        return outs


    def init_weights(self):
        if self.init_cfg:
            super().init_weights()
        else:
            kaiming_init(self.conv1.conv)
            for i in range(2, 7):
                blocks = getattr(self, f"blocks{i}")
                for block in blocks:
                    kaiming_init(block.depthwise_conv.conv)
                    kaiming_init(block.pointwise_conv.conv)
