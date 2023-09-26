### 使用说明

目前在mmdetection [v3.1.0](https://github.com/open-mmlab/mmdetection/releases/tag/v3.1.0)的基础上集成了picodet和nanodet

requirements

```
mmcv 2.0.1

mmengine 0.8.4
```



集成后在coco上进行了测试

picodet_s_320 精度

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.436
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.304
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.098
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.300
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.266
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.419
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.703
```


nanodet-plus-m_320 精度

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.269
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.411
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.281
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.080
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.275
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.456
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.255
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.392
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.417
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.130
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.472
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.673
```



### 1. 预训练权重转换

脚本在weight_conversion目录下

转换完的权重见网盘

链接: https://pan.baidu.com/s/12RuQqgeP_tD3B8h2xrjleA 提取码: x1t4 


### 2. 训练

配置文件路径在 ```mmdetection/config``` 文件夹下

#### 2.1 单卡训练

```
tools/train.py /path/to/config 
```

#### 2.2 多卡训练

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --launcher pytorch /path/to/config 
```

或者使用sh脚本

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_train.sh /path/to/config 4
```

#### 2.3 训练中断

训练如果中断， 可以使用 --resume 指定checkpoint路径 继续训练。

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/train.py --launcher pytorch /path/to/config --resume /path/to/checkpoint
```

或者在配置文件中添加

```
load_from = '/path/to/checkpoint'
resume = True
```

#### 2.4 预训练权重

##### 2.4.1 整个模型都加载

在```model```中添加```init_cfg```键, 例如

```
model = dict(
    type='PicoDet',
    init_cfg=dict(
        type='Pretrained', checkpoint='/path/to/checkpoint'),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
    ...)
```

或者在配置文件中添加

```
load_from = '/path/to/checkpoint'
```

##### 2.4.2 仅加载backbone

在```model.backbone```中添加```init_cfg```键，例如

```
model = dict(
    type='PicoDet',
    data_preprocessor=dict(
    ...
    )
    backbone=dict(
        type='LCNet',
        scale=0.75,
        feature_maps=[3, 4, 5],
        init_cfg=dict(
            type='Pretrained', checkpoint='/path/to/checkpoint')
    ),
```

上面适用于使用分类任务预训练得到的权重。如果要使用检测任务或者其他任务训练得到的权重时，权重state_dict中的key会带有backbone前缀，加载时需要去除。此时配置写法如下。

```
model = dict(
    type='PicoDet',
    data_preprocessor=dict(
    ...
    )
    backbone=dict(
        type='LCNet',
        scale=0.75,
        feature_maps=[3, 4, 5],
        init_cfg=dict(
            type='Pretrained', prefix='backbone.', checkpoint='/path/to/checkpoint')
    ),
```

#### 2.5 自定义数据集

mm框架本身支持多种格式数据集。这里只介绍coco格式。

其他格式数据集使用方法类似，建议使用时查看源码 ```mmdetection/mmdet/datasets```， 不行就转coco。

在train_dataloader.dataset中设置data_root, metainfo, ann_file以及data_prefix，例如

```
data_root = 'data/custom/'
metainfo = {
    'classes': ('classA', 'classB'),
    'palette': [
        (220, 20, 60), (119, 11, 32)
    ]

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    pin_memory=True,
    dataset=dict(
        metainfo=metainfo,
        ann_file='annotations/custom_train.json',
        data_prefix=dict(img='train/'),
        pipeline=train_pipeline))
```

metainfo中设置类别名，同时给每一类设置一个颜色。在可视化的时候会用到。

这里标注文件的路径是data_root + ann_file.

 ```data/custom/annotations/custom_train.json```

图像文件夹的路径是data_root+data_prefix['img']. 

```data/custom/train```



然后修改model.bbox_head中的类别数量。

```
model = dict(
...
	bbox_head=dict(
		...
		num_classes=2
	)
```

#### 2.6 训练日志

训练日志保存在work_dirs下。

将logs文件夹复制到本地。找一个有tensorboard的环境

```
cd logs
tensorboard --logdir . --samples_per_plugin "images=100"
```

### 3. 验证

#### 3.1 单卡验证

```
tools/test.py /path/to/config /path/to/checkpoint
```

#### 3.2 多卡验证

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 tools/test.py --launcher pytorch /path/to/config /path/to/checkpoint
```

或者使用sh脚本

```
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/dist_test.sh /path/to/config /path/to/checkpoint 4
```

### 4. 测试

```
python demo/image_demo.py /path/to/folder/or/image /path/to/config /path/to/checkpoint
```

