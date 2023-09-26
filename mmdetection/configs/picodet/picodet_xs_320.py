_base_ = './picodet_s_320.py'

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x0_35_pretrained.pth"

model = dict(
    backbone=dict(
        scale=0.35,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint)),
    neck=dict(
        in_channels=[48, 88, 176])
)