_base_ = './picodet_s_416.py'

checkpoint = "/data-nbd/mm/mmdetection/pretrained_weights/converted/lcnet/LCNet_x2_0_pretrained.pth"

model = dict(
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

max_epochs = 250

train_cfg = dict(max_epochs=max_epochs)

train_dataloader = dict(batch_size=20)
val_dataloader = dict(batch_size=20)