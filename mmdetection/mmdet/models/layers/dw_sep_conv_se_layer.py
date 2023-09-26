from mmcv.cnn import DepthwiseSeparableConvModule
from typing import Dict, Optional, Tuple, Union
import torch
from .se_layer import SELayer
from mmengine.model import bias_init_with_prob, constant_init, normal_init, kaiming_init

class DepthwiseSeparableConvSEModule(DepthwiseSeparableConvModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 se_cfg: Optional[Dict] = None,
                 dw_norm_cfg: Union[Dict, str] = 'default',
                 dw_act_cfg: Union[Dict, str] = 'default',
                 pw_norm_cfg: Union[Dict, str] = 'default',
                 pw_act_cfg: Union[Dict, str] = 'default',
                 **kwargs):
        super().__init__(in_channels,
                         out_channels,
                         kernel_size,
                         stride,
                         padding,
                         dilation,
                         norm_cfg,
                         act_cfg,
                         dw_norm_cfg,
                         dw_act_cfg,
                         pw_norm_cfg,
                         pw_act_cfg,
                         **kwargs)
        self.with_se = se_cfg is not None
        if self.with_se:
            assert isinstance(se_cfg, dict)
            self.se = SELayer(**se_cfg)

    def init_weights(self):
        kaiming_init(self.depthwise_conv.conv)
        kaiming_init(self.pointwise_conv.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        if self.with_se:
            x = self.se(x)
        x = self.pointwise_conv(x)
        return x