_base_ = './picodet_s_320.py'

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x1_5_pretrained.pth"

model = dict(
    backbone=dict(
        scale=1.5,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[192, 384, 768],
        out_channels=128),
    bbox_head=dict(
        in_channels=128,
        feat_channels=128,
        stacked_convs=4)
)

train_dataloader = dict(batch_size=40)
val_dataloader = dict(batch_size=40)