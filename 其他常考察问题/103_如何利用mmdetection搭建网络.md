总的来说，就是在cfg模型部分的配置文件写好对应的结构（backbone、neck、head等）。
然后`build_detector`就会根据cfg中的配置文件构建出对应的网络结构。

以下粘贴自博客



## 前言

这篇博客图解mmDetection搭建检测器网络架构的代码流程。在讲解的过程中，还是会以3D目标检测框架SA-SSD做例子讨论。

## 网络架构

### 2.1 理论上的说明

这段说明以单阶段3D目标检测框架SA-SSD为例。它的输入是$[W_0,L_0,H_0,C_0]$的稀疏三维特征张量。SA-SSD由backbone，neck，和detector三部分组成。它前向计算的图解如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200605163243137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NzMyNjg0,size_16,color_FFFFFF,t_70#pic_center)

SA-SSD是单阶段目标检测算法。如果这个算法扩展为多阶段，可以参考PointRCNN网络，添加一个新的环节，把3d检测框内的点云特征“收集”起来（这个收集过程记为ROI Pooling），用来回归一个更加精细的3d检测框。

### 2.2 搭建网络

mmDetection框架搭建网络的过程很好理解，主要是根据配置参数cfg调用函数build_detector：

```python
model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
```

其中`build_detector`是根据配置参数`cfg`迭代拼接网络模块的。迭代依据是`for cfg_ in cfg`。

```python
def build_detector(cfg, train_cfg=None, test_cfg=None):
    from . import detectors
    return build(cfg, detectors, dict(train_cfg=train_cfg, test_cfg=test_cfg))

def build(cfg, parrent=None, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent, default_args)

```

其中cfg模型部分的配置文件是一个字典型变量：

```python
model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='SimpleVoxel',
        num_input_features=4,
        use_norm=True,
        num_filters=[32, 64],
        with_distance=False
    ),
    neck=dict(
        type='SpMiddleFHD',
        output_shape=[40, 1600, 1408],
        num_input_features=4,
        num_hidden_features=64 * 5,
    ),
    bbox_head=dict(
        type='SSDRotateHead',
        num_class=1,
        num_output_filters=256,
        num_anchor_per_loc=2,
        use_sigmoid_cls=True,
        encode_rad_error_by_sin=True,
        use_direction_classifier=True,
        box_code_size=7,
    ),
    extra_head=dict(
        type='PSWarpHead',
        grid_offsets = (0., 40.),
        featmap_stride=.4,
        in_channels=256,
        num_class=1,
        num_parts=28,
    )
)
```

## 参考链接

[小白科研笔记：深入理解mmDetection框架——网络架构](https://blog.csdn.net/qq_39732684/article/details/106564182)

