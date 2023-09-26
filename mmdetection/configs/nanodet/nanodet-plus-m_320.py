_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py'
]

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/shufflenetv2/shufflenetv2_x1-5666bf0f80_new.pth"

model = dict(
    type='NanoDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None),
    backbone=dict(
        type='ShuffleNetV2',
        model_size='1.0x',
        out_stages=[2, 3, 4],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        type='GhostPAN',
        in_channels=[116, 232, 464],
        out_channels=96,
        num_outs=4,
        kernel_size=5,
        use_depthwise=True,
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    bbox_head=dict(
        type='NanoDetPlusHead',
        num_classes=80,
        in_channels=96,
        reg_max=7,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    aux_bbox_head=dict(
        type='SimpleConvHead',
        num_classes=80,
        in_channels=192,
        reg_max=7,
        stacked_convs=4,
        feat_channels=192,
        anchor_generator=dict(
            type='MlvlPointGenerator',
            offset=0,
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        norm_cfg=dict(type='GN', num_groups=32),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1)),
    train_cfg=dict(
        detach_epoch=10,
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13, with_center_dist_cost=False),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Albu',
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
    dict(type='Resize', scale=(320, 320), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(320, 320), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    batch_sampler=None,
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 300
base_lr = 1e-3
interval = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval)

val_evaluator = dict(proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    clip_grad=dict(type='value', clip_value=35)
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=base_lr*0.05,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True,
        convert_to_iter_based=True),
]

# hooks
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook',
                    interval=interval,
                    save_best='coco/bbox_mAP',
                    rule='greater',
                    published_keys=['meta', 'state_dict']),
    visualization=dict(type='DetVisualizationHook', draw=True, interval=5)
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=2e-4,
        update_buffers=True),
    dict(type='SetEpochInfoHook'),
]

auto_scale_lr = dict(enable=True, base_batch_size=96)
