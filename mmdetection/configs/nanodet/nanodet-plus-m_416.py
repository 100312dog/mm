_base_ = './nanodet-plus-m_320.py'

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
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(416, 416), keep_ratio=False),
    dict(type='LoadAnnotations', with_bbox=True, with_label=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=48,
    dataset=dict(pipeline=train_pipeline))

test_dataloader = dict(
    batch_size=48,
    dataset=dict(pipeline=test_pipeline))