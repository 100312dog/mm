import torch
import torch.nn as nn

from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, is_norm
from mmdet.models.layers import DepthwiseSeparableConvSEModule
from mmengine.model import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.registry import MODELS
from ..utils import make_divisible
from mmengine.model import bias_init_with_prob, constant_init, normal_init, kaiming_init

def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, norm_cfg, act_cfg):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = DepthwiseSeparableConvModule(
                inp,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_act_cfg=act_cfg,
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            ConvModule(
                inp if (self.stride > 1) else branch_features,
                branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            DepthwiseSeparableConvModule(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
                norm_cfg=norm_cfg,
                dw_act_cfg=None,
                pw_act_cfg=act_cfg,
            )
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

@MODELS.register_module()
class ShuffleNetV2(BaseModule):
    def __init__(
        self,
        model_size="1.5x",
        out_stages=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        init_cfg=None,
    ):
        super().__init__(init_cfg)
        # out_stages can only be a subset of (2, 3, 4)
        assert set(out_stages).issubset((2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        self.out_stages = out_stages
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size
        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = ConvModule(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        input_channels = output_channels
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, norm_cfg=norm_cfg, act_cfg=act_cfg
                )
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg
                    )
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = ConvModule(
                input_channels,
                output_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            )
            self.stage4.add_module("conv5", conv5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_stages:
                output.append(x)
        return tuple(output)

    def init_weights(self):
        if self.init_cfg:
            super().init_weights()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, mean=0, std=1.0 / m.weight.shape[1], bias=0)
                if is_norm(m):
                    constant_init(m, val=1, bias=1e-4)