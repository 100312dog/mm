_base_ = [
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py',
    '../_base_/datasets/coco_detection.py'
]

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x0_75_pretrained.pth"

model = dict(
    type='PicoDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        batch_augments=[
            dict(
                type='BatchSyncRandomChoiceResize',
                scales=[256, 288, 320, 352, 384],
                interval=1)
        ]),
    backbone=dict(
        type='LCNet',
        scale=0.75,
        feature_maps=[3, 4, 5],
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        type='LCPAN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_outs=4,
        kernel_size=5,
        use_depthwise=True),
    bbox_head=dict(
        type='PicoHeadV2',
        num_classes=80,
        in_channels=96,
        feat_channels=96,
        reg_max=7,
        stacked_convs=2,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            scales=[5],
            strides=[8, 16, 32, 64],
            center_offset=0.5),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='VarifocalLoss',
            use_sigmoid=True,
            alpha=0.75,
            gamma=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.5),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.5),
        use_se=True,
        use_align_head=True,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='HSwish')
    ),
    train_cfg=dict(
        initial_epoch=100,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.025,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100),
)

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(type='MinIoURandomCrop'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Resize', scale=(384, 384), keep_ratio=False),
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
    pin_memory=True,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=64, num_workers=8, dataset=dict(pipeline=test_pipeline))
test_dataloader = val_dataloader

max_epochs = 300
base_lr = 0.32
interval = 10

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=interval)

val_evaluator = dict(proposal_nums=(100, 1, 10))
test_evaluator = val_evaluator

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=4e-5),
    paramwise_cfg=dict(
        custom_keys={
            'backbone.conv1.bn.weight': dict(decay_mult=0),
            'backbone.conv1.bn.bias': dict(decay_mult=0),
            'backbone.blocks2.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks2.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks2.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks2.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks3.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks3.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks3.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks3.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks3.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks3.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks3.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks3.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks4.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks4.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks4.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks4.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks4.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks4.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks4.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks4.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.2.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.2.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.2.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.2.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.3.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.3.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.3.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.3.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.4.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.4.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.4.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.4.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.5.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.5.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks5.5.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks5.5.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks6.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks6.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks6.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks6.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks6.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks6.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'backbone.blocks6.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'backbone.blocks6.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.0.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.0.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.0.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.0.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.0.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.0.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.0.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.0.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.1.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.1.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.1.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.1.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.1.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.1.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.fpn_convs.1.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.fpn_convs.1.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.0.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.0.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.0.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.0.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.0.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.0.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.0.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.0.1.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.1.0.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.1.0.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.1.0.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.1.0.pointwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.1.1.depthwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.1.1.depthwise_conv.bn.bias': dict(decay_mult=0),
            'neck.pafpn_convs.1.1.pointwise_conv.bn.weight': dict(decay_mult=0),
            'neck.pafpn_convs.1.1.pointwise_conv.bn.bias': dict(decay_mult=0),
        })
)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=300),
    dict(
        # use cosine lr from 150 to 300 epoch
        type='CosineAnnealingLR',
        eta_min=0,
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
        update_buffers=True)
]

auto_scale_lr = dict(enable=True, base_batch_size=4*64)
