_base_ = './picodet_s_320.py'

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x2_0_pretrained.pth"

model = dict(
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='BatchSyncRandomChoiceResize',
                scales=[576, 608, 640, 672, 704],
                interval=1)
        ]),
    backbone=dict(
        scale=2.0,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[256, 512, 1024],
        out_channels=160),
    bbox_head=dict(
        in_channels=160,
        feat_channels=160,
        stacked_convs=4)
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
    dict(type='Resize', scale=(704, 704), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(640, 640), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=test_pipeline))

max_epochs = 200

train_cfg = dict(max_epochs=max_epochs)