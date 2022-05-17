# 目标检测中的mAP

## 目标检测中的精确率P、召回率R

同分类问题中的计算方法，不过这里的TP、FP的计算方法有些区别（且目标检测不存在TN）

二者的公式为：

**精确率（Precision）:**	$Precision=\frac{TP}{TP+FP}=\frac{TP}{Det}$ 

**召回率（Recall）:**	$Recall=\frac{TP}{TP+FN}=\frac{TP}{GT}$ 

其中Det为所有的**预测框个数**、GT为所有的**真实框个数**。因此计算PR曲线，只需要计算TP即可，FP=Det-TP

**TP的定义为：**

1. 预测框（Bbox）与真实框（GT）的IOU大于设定阈值**（$AP_{50}$就是0.5，$AP_{75}$就是0.75）。**
2. 置信度评分高于设定的置信度阈值（就是通过改变这个置信度阈值得到不同的PR值，画出PR曲线）

如果满足第2条，不满足第1条，那这个目标就是FP。

**当对应同一个真值有多个预测结果时，只有最高置信度分数的预测结果被认为是True Positive，其余被认为是False Positive。**

因此已知Det、GT，通过**改变置信度阈值**计算TP进而得到PR就可以画出PR曲线。

<img src="https://upload-images.jianshu.io/upload_images/10758717-2d01a412c4d0b524.gif?imageMogr2/auto-orient/strip|imageView2/2/w/480" alt="img" style="zoom:50%;" />

## AP计算

上述PR曲线有抖动，需要进行平滑处理，具体方式：**每个“峰值点”往左画一条线段直到与上一个峰值点的垂直线相交**。这样画出来的线段与坐标轴围起来的面积就是AP值。

<img src="https://upload-images.jianshu.io/upload_images/10758717-821f2ae92cb65950.gif?imageMogr2/auto-orient/strip|imageView2/2/w/640" alt="img" style="zoom:50%;" />

**AP其实是平滑后PR曲线与X轴的包络面积，而AUC是ROC曲线与X轴的包络面积，二值本质上计算方法相同**

具体的编程实现思路为：

```python
# 11个置信度阈值对应的PR值
P_list = [?]*11
R_list = [?]*11
AP = 0;
for i in range(0, 11):
    # 取R右侧最大的P计算面积
    w = R[i] - R[i-1] # 横坐标长度
    h = max(P_list[i:]) # i阈值对应的右侧最大P（包括自身）
	AP += w * h;
```



## mAP计算

mAP就是对所有类的AP值求平均。

伪代码

> input: GT、预测框（Pre）、IOU阈值（0.5）
>
> 1. 找出所有Pre中与GT的IOU大于阈值的框Pre2
>    （再检测一个GT是否对应多个Pre，如果对应多个，就把置信度最高的保留，其余的剔除）
> 2. 设置一个置信度范围，如果Pre2的置信度大于这个置信度阈值就标记为TP
> 3. 根据公式计算出当前置信度阈值下的PR值
> 4. 利用所有置信度阈值下对应的PR值计算出AP（用11点插值法求PR曲线面积即可）
> 5. 所有类别的AP求平均就是当前IOU阈值下的$mAP@0.5$

### 代码实现

```cpp
# 按照置信度降序排序
sorted_ind = np.argsort(-confidence)
BB = BB[sorted_ind, :]   # 预测框坐标
image_ids = [image_ids[x] for x in sorted_ind] # 各个预测框的对应图片id

# 便利预测框，并统计TPs和FPs
nd = len(image_ids)
tp = np.zeros(nd)
fp = np.zeros(nd)
for d in range(nd):
    R = class_recs[image_ids[d]]
    bb = BB[d, :].astype(float)
    ovmax = -np.inf
    BBGT = R['bbox'].astype(float)  # ground truth

    if BBGT.size > 0:
        # 计算IoU
        # intersection
        ixmin = np.maximum(BBGT[:, 0], bb[0])
        iymin = np.maximum(BBGT[:, 1], bb[1])
        ixmax = np.minimum(BBGT[:, 2], bb[2])
        iymax = np.minimum(BBGT[:, 3], bb[3])
        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)
        inters = iw * ih

        # union
        uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
               (BBGT[:, 2] - BBGT[:, 0] + 1.) *
               (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

        overlaps = inters / uni
        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)
    # 取最大的IoU
    if ovmax > ovthresh:  # 是否大于阈值
        if not R['difficult'][jmax]:  # 非difficult物体
            if not R['det'][jmax]:    # 未被检测
                tp[d] = 1.
                R['det'][jmax] = 1    # 标记已被检测
            else:
                fp[d] = 1.
    else:
        fp[d] = 1.

# 计算precision recall
fp = np.cumsum(fp)
tp = np.cumsum(tp)
rec = tp / float(npos)
# avoid divide by zero in case the first detection matches a difficult
# ground truth
prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
```

这里最终得到一系列的precision和recall值，并且这些值是按照置信度降低排列统计的，可以认为是取不同的置信度阈值（或者rank值）得到的。

```cpp
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
```

计算各个类别的AP值后，取平均值就可以得到最终的mAP值了。但是对于COCO数据集相对比较复杂，不过其提供了计算的API，感兴趣可以看一下[cocodataset/cocoapi](https://link.zhihu.com/?target=https%3A//github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)。

## 参考链接

[目标检测算法的评估指标：mAP定义及计算方式](https://www.jianshu.com/p/fd9b1e89f983)（特别详细，但是有点云里雾里）

[理解目标检测当中的mAP](https://blog.csdn.net/hsqyc/article/details/81702437)（以实际例子展示如何计算mAP，很容易理解）

# ReID的评价指标

## rank-n

**搜索结果中最靠前（置信度最高）的n张图有正确结果的概率。**

只要存在正确结果就是100%，不存在就是0%

## mAP

重点了解TP的计算方法：

1. 假设待查询图片一共有m个目标，取检测结果的前n个
2. TP：**这n个结果中，匹配正确的目标数**

则$P=\frac{TP}{n},R=\frac{TP}{m}$，注意这里的R不用，**只使用P计算AP值**。**而F1-score是使用PR的**，计算方法也与其他一样。

<img src="https://img2018.cnblogs.com/blog/1294177/201908/1294177-20190812180434701-1553363434.png" alt="img" style="zoom:80%;" />

上面这个表述有问题，每个P的n是不一样的，具体计算看下图

mAP就是所有待查询结果的平均值

<img src="https://img2018.cnblogs.com/blog/1294177/201908/1294177-20190812191206432-1535540428.png" alt="img" style="zoom:100%;" />

## 参考链接

[行人重识别和车辆重识别（ReID）中的评测指标——mAP和Rank-k](https://www.cnblogs.com/tay007/p/11341701.html)（mAP介绍的很好）

[PRID：行人重识别常用评测指标（rank-n、Precision & Recall、F-score、mAP 、CMC、ROC)](https://blog.csdn.net/qq_41978139/article/details/106993152)（很全，但文章排版不好）

# 目标跟踪的评价指标

## MOTA

## MOTP

# 扩展

准确率acc是：预测为正，实际为正和预测为负，实际负占总样本的比例。与精确率P是不同的。

## COCO的AP和AR

上述的的mAP是VOC的计算方法，10年之后大都使用COCO的评估标准。

**<font color='red'><big>AP</big></font>**

①COCO中的AP@.5就是VOC的mAP
②而COCO的mAP是VOCmAP(0.05:0.5:0.95)的平均值，即在类别和IOU上的平均。
评测更加全面

用于判断查找的目标框是否足够准确（查准率）

**<font color='red'><big>AR</big></font>**

$AR_{100}$：选择100个检测结果，该结果对应的**最大召回率**，然后在**所有的类别和所有IOU**上求平均（同mAP的计算范围）。

最大召回率的含义：举例，GT有100个，检测结果中在IOU阈值为0.5时有20个认为是TP，那么他的$AR_{50}@0.5=R=20/100=0.5$​​，即选取使召回率最大的20个检测框来计算这个最大的召回率。

### 参考链接

[COCO目标检测测评指标](https://www.jianshu.com/p/d7a06a720a2b)

## 一些问题

**为什么目标检测算法要选取AP或者mAP来评估性能优劣？而不是准确率(accuracy)/精确率(precision)/召回率(recall)？**

如果只关注单一指标，可能会导致一个指标精度很高，其他指标很低，没有办法比较这些算法的优劣，需要综合考虑多个性能指标才能更加准确的反映算法的好坏。也就是既要获取较高的精确率也要获取较高的召回率。这时可以考虑在二维图上很方便绘制出PR曲线，每种算法每种检测的目标类型都可以绘制独立的一条曲线，但问题是单纯看曲线也很难比较算法的综合优劣，故而考虑曲线下的面积AP来计算，面积越接近1性能越好。曲线下的面积理解为不同召回值的情况下所有精度的平均值。