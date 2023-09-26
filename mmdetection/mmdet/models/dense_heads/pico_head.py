# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union, Sequence

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, is_norm
from mmengine import MessageHub
from mmengine.model import bias_init_with_prob, constant_init, normal_init, kaiming_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import distance2bbox
from mmdet.utils import ConfigType, InstanceList, OptInstanceList, MultiConfig, reduce_mean
from ..task_modules import anchor_inside_flags
from ..utils import (filter_scores_and_topk, images_to_levels, multi_apply, sigmoid_geometric_mean,
                     unmap)
from .atss_head import ATSSHead
from .gfl_head import Integral
from ..layers.transformer import inverse_sigmoid
from mmengine.model import BaseModule
from mmdet.models.layers import ChannelAttention
import torch.nn.functional as F
from mmdet.structures.bbox import bbox_overlaps
from typing import Dict, Optional, Tuple, Union


class DPModule(BaseModule):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Dict = dict(type='ReLU'),
                 dw_norm_cfg: Union[Dict, str] = 'default',
                 dw_act_cfg: Union[Dict, str] = 'default',
                 pw_norm_cfg: Union[Dict, str] = 'default',
                 pw_act_cfg: Union[Dict, str] = 'default',
                 **kwargs):
        super().__init__()
        assert 'groups' not in kwargs, 'groups should not be specified'

        # if norm/activation config of depthwise/pointwise ConvModule is not
        # specified, use default config.
        dw_norm_cfg = dw_norm_cfg if dw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        dw_act_cfg = dw_act_cfg if dw_act_cfg != 'default' else act_cfg
        pw_norm_cfg = pw_norm_cfg if pw_norm_cfg != 'default' else norm_cfg  # type: ignore # noqa E501
        pw_act_cfg = pw_act_cfg if pw_act_cfg != 'default' else act_cfg

        # depthwise convolution
        self.depthwise_conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            norm_cfg=dw_norm_cfg,  # type: ignore
            act_cfg=dw_act_cfg,  # type: ignore
            **kwargs)

        self.pointwise_conv = ConvModule(
            out_channels,
            out_channels,
            1,
            norm_cfg=pw_norm_cfg,  # type: ignore
            act_cfg=pw_act_cfg,  # type: ignore
            **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class PicoSE(BaseModule):
    def __init__(self,
                 feat_channels,
                 norm_cfg,
                 act_cfg: MultiConfig = (dict(type='Sigmoid'),
                                         dict(type='HSwish'))
                 ):
        super(PicoSE, self).__init__()
        self.attention = ChannelAttention(feat_channels, act_cfg=act_cfg[0])
        self.conv = ConvModule(feat_channels, feat_channels, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.attention(x)
        out = self.conv(out)
        return out


@MODELS.register_module()
class PicoHeadV2(ATSSHead):
    """
    PicoHeadV2
    Args:
        conv_feat (object): Instance of 'PicoFeat'
        num_classes (int): Number of classes
        fpn_stride (list): The stride of each FPN Layer
        prior_prob (float): Used to set the bias init for the class prediction layer
        loss_class (object): Instance of VariFocalLoss.
        loss_dfl (object): Instance of DistributionFocalLoss.
        loss_bbox (object): Instance of bbox loss.
        assigner (object): Instance of label assigner.
        reg_max: Max value of integral set :math: `{0, ..., reg_max}`
                n QFL setting. Default: 7.
    """

    def __init__(self,
                 reg_max: int = 7,
                 use_se: bool = True,
                 use_align_head: bool = True,
                 act_cfg=dict(type='HSwish'),
                 loss_dfl: ConfigType = dict(
                     type='DistributionFocalLoss', loss_weight=0.25),
                 **kwargs) -> None:

        self.act_cfg = act_cfg
        self.use_se = use_se
        self.use_align_head = use_align_head
        self.reg_max = reg_max

        super().__init__(**kwargs)

        self.integral = Integral(self.reg_max)
        self.loss_dfl = MODELS.build(loss_dfl)

        if self.train_cfg:
            self.initial_epoch = self.train_cfg['initial_epoch']
            self.initial_assigner = TASK_UTILS.build(
                self.train_cfg['initial_assigner'])
            self.assigner = self.initial_assigner
            self.alignment_assigner = TASK_UTILS.build(
                self.train_cfg['assigner'])
            self.alpha = self.train_cfg['alpha']
            self.beta = self.train_cfg['beta']

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()

        if self.use_se:
            self.se = nn.ModuleList()

        for n in range(len(self.prior_generator.strides)):
            cls_subnet_convs = nn.ModuleList()
            for i in range(self.stacked_convs):
                chn = self.in_channels if i == 0 else self.feat_channels
                cls_subnet_convs.add_module('cls_conv_dw{}'.format(i),
                                            ConvModule(chn,
                                                       chn,
                                                       5,
                                                       1,
                                                       padding=2,
                                                       groups=chn,
                                                       norm_cfg=self.norm_cfg,
                                                       act_cfg=self.act_cfg)
                                            )
                cls_subnet_convs.add_module('cls_conv_pw{}'.format(i),
                                            ConvModule(chn,
                                                       self.feat_channels,
                                                       1,
                                                       1,
                                                       norm_cfg=self.norm_cfg,
                                                       act_cfg=self.act_cfg)
                                            )
            self.cls_convs.append(cls_subnet_convs)

            if self.use_se:
                self.se.append(PicoSE(self.feat_channels, norm_cfg=self.norm_cfg, act_cfg=(dict(type='Sigmoid'), self.act_cfg)))


        self.pico_cls = nn.ModuleList()
        self.pico_reg = nn.ModuleList()

        if self.use_align_head:
            self.pico_cls_align = nn.ModuleList()

        for i in range(len(self.prior_generator.strides)):
            self.pico_cls.append(
                nn.Conv2d(
                    in_channels=self.feat_channels,
                    out_channels=self.cls_out_channels,
                    kernel_size=1,
                    stride=1)
            )
            self.pico_reg.append(
                nn.Conv2d(
                    in_channels=self.feat_channels,
                    out_channels=4 * (self.reg_max + 1),
                    kernel_size=1,
                    stride=1)
            )
            if self.use_align_head:
                self.pico_cls_align.append(
                    DPModule(
                        self.feat_channels,
                        1,
                        5,
                        padding=2,
                        norm_cfg=self.norm_cfg,
                        dw_act_cfg=self.act_cfg,
                        pw_act_cfg=None
                    ))

    def forward(self, feats: Tuple[Tensor, ...]) -> tuple:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
            - cls_scores (list[Tensor]): Classification scores for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for all scale
              levels, each is a 4D-tensor, the channels number is
              num_base_priors * 4.
        """

        cls_scores = []
        bbox_preds = []
        for idx, (x, stride) in enumerate(
                zip(feats, self.prior_generator.strides)):
            cls_feat = x

            for cls_layer in self.cls_convs[idx]:
                cls_feat = cls_layer(cls_feat)

            reg_feat = cls_feat

            if self.use_se:
                reg_feat = self.se[idx](cls_feat)

            cls_score = self.pico_cls[idx](reg_feat)
            reg_pred = self.pico_reg[idx](reg_feat)

            if self.use_align_head:
                cls_score = inverse_sigmoid(
                    sigmoid_geometric_mean(cls_score, self.pico_cls_align[idx](cls_feat)))

            cls_scores.append(cls_score)
            bbox_preds.append(reg_pred)

        return tuple(cls_scores), tuple(bbox_preds)

    def init_weights(self) -> None:
        """Initialize weights of the head."""
        bias_cls = bias_init_with_prob(0.01)
        for cls_layer in self.pico_cls:
            normal_init(cls_layer, std=0.01, bias=bias_cls)
        for reg_layer in self.pico_reg:
            normal_init(reg_layer, std=0.01)
        if self.use_align_head:
            for cls_align_layer in self.pico_cls_align:
                kaiming_init(cls_align_layer.depthwise_conv.conv, distribution="uniform")
                kaiming_init(cls_align_layer.pointwise_conv.conv, distribution="uniform")
        if self.use_se:
            for se_layer in self.se:
                normal_init(se_layer.attention.fc, std=0.001)

    def get_targets(self,
                    cls_scores: List[List[Tensor]],
                    bbox_preds: List[List[Tensor]],
                    anchor_list: List[List[Tensor]],
                    valid_flag_list: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    unmap_outputs: bool = True) -> tuple:
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            cls_scores (list[list[Tensor]]): Classification predictions of
                images, a 3D-Tensor with shape [num_imgs, num_priors,
                num_classes].
            bbox_preds (list[list[Tensor]]): Decoded bboxes predictions of one
                image, a 3D-Tensor with shape [num_imgs, num_priors, 4] in
                [tl_x, tl_y, br_x, br_y] format.
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 4).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: a tuple containing learning targets.

                - anchors_list (list[list[Tensor]]): Anchors of each level.
                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each
                  level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - norm_alignment_metrics_list (list[Tensor]): Normalized
                  alignment metrics of each level.
        """
        num_imgs = len(batch_img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_imgs
        # anchor_list: list(b * [-1, 4])

        # get epoch information from message hub
        message_hub = MessageHub.get_current_instance()
        self.epoch = message_hub.get_info('epoch')

        if self.epoch < self.initial_epoch:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_bbox_weights, pos_inds_list, neg_inds_list,
             sampling_result) = multi_apply(
                 super()._get_targets_single,
                 anchor_list,
                 valid_flag_list,
                 num_level_anchors_list,
                 batch_gt_instances,
                 batch_img_metas,
                 batch_gt_instances_ignore,
                 unmap_outputs=unmap_outputs)
            all_assign_metrics = [
                weight[..., 0] for weight in all_bbox_weights
            ]
        else:
            (all_anchors, all_labels, all_label_weights, all_bbox_targets,
             all_assign_metrics) = multi_apply(
                 self._get_targets_single,
                 cls_scores,
                 bbox_preds,
                 anchor_list,
                 valid_flag_list,
                 num_level_anchors_list,
                 batch_gt_instances,
                 batch_img_metas,
                 batch_gt_instances_ignore,
                 unmap_outputs=unmap_outputs)

        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        norm_alignment_metrics_list = images_to_levels(all_assign_metrics,
                                                       num_level_anchors)

        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, norm_alignment_metrics_list)

    def _get_targets_single(self,
                            cls_scores: Tensor,
                            bbox_preds: Tensor,
                            flat_anchors: Tensor,
                            valid_flags: Tensor,
                            num_level_anchors: List[int],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            unmap_outputs: bool = True) -> tuple:
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            cls_scores (Tensor): Box scores for each image.
            bbox_preds (Tensor): Box energies / deltas for each image.
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for current image.
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                norm_alignment_metrics (Tensor): Normalized alignment metrics
                    of all priors in the image with shape (N,).
        """
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg['allowed_border'])
        if not inside_flags.any():
            raise ValueError(
                'There is no valid anchor inside the image boundary. Please '
                'check the image size and anchor sizes, or set '
                '``allowed_border`` to -1 to skip the condition.')
        # assign gt and sample anchors

        anchors = flat_anchors[inside_flags, :]
        pred_instances = InstanceData(
            priors=anchors,
            scores=cls_scores[inside_flags, :],
            bboxes=bbox_preds[inside_flags, :])
        assign_result = self.alignment_assigner.assign(pred_instances,
                                                       gt_instances,
                                                       gt_instances_ignore,
                                                       self.alpha, self.beta)
        assign_ious = assign_result.max_overlaps
        assign_metrics = assign_result.assign_metrics

        sampling_result = self.sampler.sample(assign_result, pred_instances,
                                              gt_instances)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)
        norm_alignment_metrics = anchors.new_zeros(
            num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            # point-based
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets

            labels[pos_inds] = sampling_result.pos_gt_labels
            if self.train_cfg['pos_weight'] <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg['pos_weight']
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        class_assigned_gt_inds = torch.unique(
            sampling_result.pos_assigned_gt_inds)
        for gt_inds in class_assigned_gt_inds:
            gt_class_inds = pos_inds[sampling_result.pos_assigned_gt_inds ==
                                     gt_inds]
            pos_alignment_metrics = assign_metrics[gt_class_inds]
            pos_ious = assign_ious[gt_class_inds]
            pos_norm_alignment_metrics = pos_alignment_metrics / (
                pos_alignment_metrics.max() + 10e-8) * pos_ious.max()
            norm_alignment_metrics[gt_class_inds] = pos_norm_alignment_metrics

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            norm_alignment_metrics = unmap(norm_alignment_metrics,
                                           num_total_anchors, inside_flags)
        return (anchors, labels, label_weights, bbox_targets,
                norm_alignment_metrics)


    def loss_by_feat_single(self, anchors: Tensor, cls_score: Tensor,
                            bbox_pred: Tensor, decode_bbox_pred: Tensor,
                            labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            alignment_metrics: Tensor,
                            stride: Tuple[int, int]) -> dict:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Decoded bboxes for each scale
                level with shape (N, num_anchors * 4, H, W).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors).
            bbox_targets (Tensor): BBox regression targets of each anchor with
                shape (N, num_total_anchors, 4).
            alignment_metrics (Tensor): Alignment metrics with shape
                (N, num_total_anchors).
            stride (Tuple[int, int]): Downsample stride of the feature map.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.reshape(-1, self.cls_out_channels)
        bbox_pred = bbox_pred.reshape(-1, 4 * (self.reg_max + 1))
        decode_bbox_pred = decode_bbox_pred.reshape(-1, 4)
        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        alignment_metrics = alignment_metrics.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_pred = decode_bbox_pred[pos_inds]

            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors)
            target_corners = self.bbox_coder.encode(pos_anchor_centers / stride[0],
                                                    pos_bbox_targets / stride[0],
                                                    self.reg_max).reshape(-1)
            pos_bbox_pred = bbox_pred[pos_inds]
            pred_corners = pos_bbox_pred.reshape(-1, self.reg_max + 1)

            # get epoch information from message hub
            message_hub = MessageHub.get_current_instance()
            self.epoch = message_hub.get_info('epoch')
            if self.epoch < self.initial_epoch:
                alignment_metrics[pos_inds] *= bbox_overlaps(pos_decode_bbox_pred.detach(), pos_bbox_targets, is_aligned=True)

            pos_bbox_weight = alignment_metrics[pos_inds]
            # regression loss
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_bbox_targets,
                weight=pos_bbox_weight,
                avg_factor=1.0)

            # dfl loss
            loss_dfl = self.loss_dfl(
                pred_corners,
                target_corners,
                weight=pos_bbox_weight[:, None].expand(-1, 4).reshape(-1),
                avg_factor=4.0)

        else:
            loss_bbox = bbox_pred.sum() * 0
            loss_dfl = bbox_pred.sum() * 0
            pos_bbox_weight = bbox_targets.new_tensor(0.)

        targets = F.one_hot(labels, num_classes=self.num_classes + 1)[:, :self.num_classes] * alignment_metrics[:, None]
        loss_cls = self.loss_cls(
            cls_score, targets, label_weights, avg_factor=1.0)

        return loss_cls, loss_bbox, loss_dfl, \
               alignment_metrics.sum(), pos_bbox_weight.sum()

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Decoded box for each scale
                level with shape (N, num_anchors * 4, H, W) in
                [tl_x, tl_y, br_x, br_y] format.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], Optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(batch_img_metas)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        flatten_cls_scores_list = [cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
                                   for cls_score in cls_scores]

        flatten_bbox_preds_list = [bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4 * (self.reg_max + 1))
                                   for bbox_pred in bbox_preds]

        flatten_decode_bbox_preds_list = []
        for flatten_bbox_pred, anchors, stride in zip(flatten_bbox_preds_list, anchor_list[0], self.prior_generator.strides):
            flatten_bbox_pred_corners = self.integral(flatten_bbox_pred) * stride[0]
            flatten_decoded_bbox_pred = self.bbox_coder.decode(self.anchor_center(anchors).repeat((num_imgs,1)),
                                                       flatten_bbox_pred_corners)
            flatten_decode_bbox_preds_list.append(flatten_decoded_bbox_pred.reshape(num_imgs, -1, 4))

        cls_reg_targets = self.get_targets(
            torch.cat(flatten_cls_scores_list, dim=1).sigmoid().detach(),
            torch.cat(flatten_decode_bbox_preds_list, dim=1).detach(),
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         alignment_metrics_list) = cls_reg_targets

        losses_cls, losses_bbox, losses_dfl, \
        cls_avg_factors, bbox_avg_factors = multi_apply(
                self.loss_by_feat_single,
                anchor_list,
                flatten_cls_scores_list,
                flatten_bbox_preds_list,
                flatten_decode_bbox_preds_list,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                alignment_metrics_list,
                self.prior_generator.strides)

        cls_avg_factor = reduce_mean(sum(cls_avg_factors)).clamp_(min=1).item()
        losses_cls = list(map(lambda x: x / cls_avg_factor, losses_cls))

        bbox_avg_factor = reduce_mean(
            sum(bbox_avg_factors)).clamp_(min=1).item()
        losses_bbox = list(map(lambda x: x / bbox_avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / bbox_avg_factor, losses_dfl))
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    def anchor_center(self, anchors: Tensor) -> Tensor:
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), ``xyxy`` format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), ``xy`` format.
        """
        anchors_cx = (anchors[..., 2] + anchors[..., 0]) / 2
        anchors_cy = (anchors[..., 3] + anchors[..., 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigType,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. GFL head does not need this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (:obj: `ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
            is False and mlvl_score_factor is None, return mlvl_bboxes and
            mlvl_scores, else return mlvl_bboxes, mlvl_scores and
            mlvl_score_factor. Usually with_nms is False is used for aug
            test. If with_nms is True, then return the following format

            - det_bboxes (Tensor): Predicted bboxes with shape
              [num_bboxes, 5], where the first 4 columns are bounding
              box positions (tl_x, tl_y, br_x, br_y) and the 5-th
              column are scores between 0 and 1.
            - det_labels (Tensor): Predicted labels of the corresponding
              box with shape [num_bboxes].
        """
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_labels = []
        for level_idx, (cls_score, bbox_pred, stride, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list,
                    self.prior_generator.strides, mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert stride[0] == stride[1]

            bbox_pred = bbox_pred.permute(1, 2, 0)
            bbox_pred = self.integral(bbox_pred) * stride[0]

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()

            # After https://github.com/open-mmlab/mmdetection/pull/6268/,
            # this operation keeps fewer bboxes under the same `nms_pre`.
            # There is no difference in performance for most models. If you
            # find a slight drop in performance, you can set a larger
            # `nms_pre` than before.
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            scores, labels, _, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']

            bboxes = self.bbox_coder.decode(
                self.anchor_center(priors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

        results = InstanceData()
        results.bboxes = torch.cat(mlvl_bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)

        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)







