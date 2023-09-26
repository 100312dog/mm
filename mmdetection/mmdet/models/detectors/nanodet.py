# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

import torch
from mmengine.dist import get_world_size
from mmengine.logging import print_log
from mmengine import MessageHub

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import copy


@MODELS.register_module()
class NanoDet(SingleStageDetector):
    """Implementation of RTMDet.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
        use_syncbn (bool): Whether to use SyncBatchNorm. Defaults to True.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 aux_bbox_head: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 use_syncbn: bool = True) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        if aux_bbox_head:
            self.aux_neck = copy.deepcopy(self.neck)
            aux_bbox_head.update(train_cfg=train_cfg)
            self.aux_bbox_head = MODELS.build(aux_bbox_head)
            self.detach_epoch = self.train_cfg['detach_epoch']

        # TODO： Waiting for mmengine support
        if use_syncbn and get_world_size() > 1:
            torch.nn.SyncBatchNorm.convert_sync_batchnorm(self)
            print_log('Using SyncBatchNorm()', 'current')

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def with_aux_neck(self) -> bool:
        """bool: whether the detector has a aux_neck"""
        return hasattr(self, 'aux_neck') and self.aux_neck is not None

    @property
    def with_aux_bbox_head(self) -> bool:
        """bool: whether the detector has a aux_neck"""
        return hasattr(self, 'aux_bbox_head') and self.aux_bbox_head is not None

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        feat = self.neck(x)
        if self.with_aux_neck and self.training:
            if self.epoch < self.detach_epoch:
                aux_feat = self.aux_neck(x)
                dual_feat = (torch.cat([f, aux_f], dim=1) for f, aux_f in zip(feat, aux_feat))
            else:
                x_detach = [xx.detach() for xx in x]
                feat_detach = [f.detach() for f in feat]
                aux_feat = self.aux_neck(x_detach)
                dual_feat = (torch.cat([f, aux_f], dim=1) for f, aux_f in zip(feat_detach, aux_feat))
            return feat, dual_feat
        else:
            return feat

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        extracted_feat = self.extract_feat(batch_inputs)
        if self.with_aux_bbox_head:
            feat, dual_feat = extracted_feat
            aux_losses, aux_cls_reg_targets = self.aux_bbox_head.loss(dual_feat, batch_data_samples, with_cls_reg_targets=True)
            aux_losses = {k + "_aux": v for k, v in aux_losses.items()}
            losses = self.bbox_head.loss(feat, batch_data_samples, aux_cls_reg_targets)
            losses.update(aux_losses)
        else:
            losses = self.bbox_head.loss(extracted_feat, batch_data_samples)
        return losses
