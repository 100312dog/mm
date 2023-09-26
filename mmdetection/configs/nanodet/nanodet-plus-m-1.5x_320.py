_base_ = './nanodet-plus-m_320.py'

model = dict(
    backbone=dict(
        model_size='1.5x',
        init_cfg=None
    ),
    neck=dict(
        in_channels=[176, 352, 704],
        out_channels=128
    ),
    bbox_head=dict(
        in_channels=128,
        feat_channels=128
    ),
    aux_bbox_head=dict(
        in_channels=256,
        feat_channels=256
    )
)