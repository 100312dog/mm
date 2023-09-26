from mmengine.config import read_base
with read_base():
    from .._base_.default_runtime_pure_python import *
    from .._base_.schedules.schedule_1x_pure_python import *
    from .._base_.datasets.coco_detection_pure_python import *

from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.detectors import NanoDet
from mmdet.models.backbones.shufflenetv2 import ShuffleNetV2
from mmdet.models.necks.ghost_pan import GhostPAN
from mmdet.models.dense_heads.nanodet_plus_head import NanoDetPlusHead
from mmdet.models.dense_heads.nanodet_plus_head import SimpleConvHead
from mmdet.models.task_modules.prior_generators.point_generator import MlvlPointGenerator
from mmdet.models.task_modules.coders.distance_point_bbox_coder import DistancePointBBoxCoder
from mmdet.models.task_modules.assigners.dynamic_soft_label_assigner import DynamicSoftLabelAssigner

from mmdet.models.losses.gfocal_loss import QualityFocalLoss
from mmdet.models.losses.gfocal_loss import DistributionFocalLoss
from mmdet.models.losses.iou_loss import GIoULoss


from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.normalization import GroupNorm
from torch.nn.modules.activation import LeakyReLU


from mmengine.model.weight_init import PretrainedInit

from mmcv.transforms.loading import LoadImageFromFile
from mmdet.datasets.transforms.loading import LoadAnnotations
from mmdet.datasets.transforms.transforms import RandomFlip
from mmdet.datasets.transforms.transforms import Albu
from mmdet.datasets.transforms.transforms import Resize
from mmdet.datasets.transforms.formatting import PackDetInputs

from mmengine.optim.optimizer.optimizer_wrapper import OptimWrapper
from torch.optim.adamw import AdamW
from mmengine.optim.scheduler import LinearLR
from mmengine.optim.scheduler import CosineAnnealingLR

from mmdet.engine.hooks.visualization_hook import DetVisualizationHook
from mmdet.engine.hooks.set_epoch_info_hook import SetEpochInfoHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.ema_hook import EMAHook



checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/shufflenetv2/shufflenetv2_x1-5666bf0f80_new.pth"

model = dict(
    type=NanoDet,
    data_preprocessor=dict(
        type=DetDataPreprocessor,
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type=ShuffleNetV2,
        model_size='1.0x',
        out_stages=[2, 3, 4],
        act_cfg=dict(type=LeakyReLU, negative_slope=0.1),
        init_cfg=dict(
            type=PretrainedInit, checkpoint=checkpoint)),
    neck=dict(
        type=GhostPAN,
        in_channels=[116, 232, 464],
        out_channels=96,
        num_outs=4,
        kernel_size=5,
        use_depthwise=True,
        act_cfg=dict(type=LeakyReLU, negative_slope=0.1)),
    bbox_head=dict(
        type=NanoDetPlusHead,
        num_classes=80,
        in_channels=96,
        reg_max=7,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type=MlvlPointGenerator,
            offset=0,
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss,
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
        loss_dfl=dict(type=DistributionFocalLoss, loss_weight=0.25),
        norm_cfg=dict(type=BatchNorm2d),
        act_cfg=dict(type=LeakyReLU, negative_slope=0.1)),
    aux_bbox_head=dict(
        type=SimpleConvHead,
        num_classes=80,
        in_channels=192,
        reg_max=7,
        stacked_convs=4,
        feat_channels=192,
        anchor_generator=dict(
            type=MlvlPointGenerator,
            offset=0,
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(type=DistancePointBBoxCoder),
        loss_cls=dict(
            type=QualityFocalLoss,
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type=GIoULoss, loss_weight=2.0),
        loss_dfl=dict(type=DistributionFocalLoss, loss_weight=0.25),
        norm_cfg=dict(type=GroupNorm, num_groups=32),
        act_cfg=dict(type=LeakyReLU, negative_slope=0.1)),
    train_cfg=dict(
        detach_epoch=10,
        assigner=dict(type=DynamicSoftLabelAssigner, topk=13, with_center_dist_cost=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.025,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
)

train_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=LoadAnnotations, with_bbox=True, with_label=True),
    dict(type=RandomFlip, prob=0.5),
    dict(type=Albu,
         transforms=[
             dict(type='Affine',
                  scale=(0.6 * 0.8, 1.4 * 1.2),
                  translate_percent=(-0.2, 0.2),
                  always_apply=True),
             dict(
                 type='ColorJitter',
                 hue=0,
                 saturation=(0.5, 1.2),
                 contrast=0.4,
                 brightness=0.2,
                 always_apply=True)
         ],
         bbox_params=dict(type='BboxParams',
                          format='pascal_voc',
                          label_fields=['gt_bboxes_labels', 'gt_ignore_flags'])),
    dict(type=Resize, scale=(320, 320), keep_ratio=False),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(320, 320), keep_ratio=False),
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader.update(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader.update(
    batch_size=64, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 300
base_lr = 1e-3
interval = 10

train_cfg.update(
    max_epochs=max_epochs,
    val_interval=interval)

val_evaluator.update(proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

# optimizer
optim_wrapper.pop('optimizer')
optim_wrapper.update(
    type=OptimWrapper,
    optimizer=dict(type=AdamW, lr=base_lr, weight_decay=0.05),
    clip_grad=dict(type='value', clip_value=35)
)

# learning rate
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        # use cosine lr from 150 to 300 epoch
        type=CosineAnnealingLR,
        eta_min=base_lr*0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks.update(
    checkpoint=dict(type=CheckpointHook,
                    interval=interval,
                    save_best='coco/bbox_mAP',
                    rule='greater',
                    published_keys=['meta', 'state_dict']),
    visualization=dict(type=DetVisualizationHook, draw=True, interval=5)
)

custom_hooks = [
    dict(
        type=EMAHook,
        ema_type='ExpMomentumEMA',
        momentum=2e-4,
        update_buffers=True),
    dict(type=SetEpochInfoHook),
]

auto_scale_lr.update(enable=True, base_batch_size=96)
