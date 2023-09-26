_base_ = './picodet_s_320.py'

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x0_75_pretrained.pth"

model = dict(
    data_preprocessor=dict(
        batch_augments=[
            dict(
                type='BatchSyncRandomChoiceResize',
                scales=[352, 384, 416, 448, 480],
                interval=1)
        ]),
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
    dict(type='Resize', scale=(480, 480), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=40,
    dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=40,
    dataset=dict(pipeline=test_pipeline))