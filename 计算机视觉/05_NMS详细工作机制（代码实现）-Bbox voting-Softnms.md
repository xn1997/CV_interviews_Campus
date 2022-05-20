## 问题

看到一句话：NMS都不懂，还做什么Detection !　虎躯一震……懂是大概懂，但代码能写出来吗？？？

在目标检测网络中，产生 proposal 后使用分类分支给出每个框的每类置信度，使用回归分支修正框的位置，最终会使用 NMS 方法去除**同个类别**当中 IOU 重叠度较高且 scores 即置信度较低的那些检测框。

下图就是在目标检测中 NMS 的使用效果：emmm大概就是能让你更无遮挡地看到美女的脸吧hhhh

![《手撕非极大值抑制算法NMS与soft-NMS》](https://raw.githubusercontent.com/xn1997/picgo/master/gtXrQCvwy25bOhM.jpg)

## 背景知识

NMS (Non-maximum suppression) 非极大值抑制，即抑制不是极大值的检测框，根据什么去抑制？在目标检测领域，当然是根据 IOU (Intersection over Union) 去抑制。下图是绿色检测框与红色检测框的 IOU 计算方法：

![img](https://raw.githubusercontent.com/xn1997/picgo/master/GHivYzUtdSJrZNE.png)



## NMS 原理及示例

注意 NMS 是针对一个特定的类别进行操作的。例如假设一张图中有要检测的目标有“人脸”和“猫”，没做NMS之前检测到10个目标框，每个目标框变量表示为: $[x_1,y_1,x_2,y_2,score_1,score_2]$ ，其中 $(x_1,y_1)$ 表示该框左上角坐标，$(x_2,y_2)$ 表示该框右下角坐标，$score_1$ 表示"人脸"类别的置信度，$score_2$ 表示"猫"类别的置信度。当 $score_1$ 比 $score_2$ 大时，将该框归为“人脸”类别，反之归为“猫”类别。最后我们假设10个目标框中有6个被归类为“人脸”类别。

接下来演示如何对“人脸”类别的目标框进行 NMS 。

首先对6个目标框按照 $score_1$ 即置信度降序排序：

| 目标框 | score_1 |
| :----: | :-----: |
|   A    |   0.9   |
|   B    |  0.85   |
|   C    |   0.7   |
|   D    |   0.6   |
|   E    |   0.4   |
|   F    |   0.1   |

(1) 取出最大置信度的那个目标框 A 保存下来
(2) 分别判断 B-F 这５个目标框与 A 的重叠度 IOU ，如果 IOU 大于我们预设的阈值（一般为 0.5），则将该目标框丢弃。假设此时丢弃的是 C和 F 两个目标框，这时候该序列中只剩下 B D E 这三个。
(3) 重复以上流程，直至排序序列为空。

## 代码实现

### numpy矩阵优化版本

```python
# bboxees维度为 [N, 4]，scores维度为 [N, 1]，均为np.array()
def single_nms(self, bboxes, scores, thresh = 0.5):
    # x1、y1、x2、y2以及scores赋值
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    
    # 计算每个检测框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) 
    
    # 按照 scores 置信度降序排序, order 为排序的索引
    order = scores.argsort() # argsort为python中的排序函数，默认升序排序
    order = order[::-1] # 将升序结果翻转为降序
    
    # 保留的结果框索引
    keep = []
    
    # torch.numel() 返回张量元素个数
    while order.size > 0:
        if order.size == 1:
            i = order[0]
            keep.append(i)
            break
        else:
            i = order[0]  # 在pytorch中使用item()来取出元素的实值，即若只是 i = order[0]，此时的 i 还是一个 tensor，因此不能赋值给 keep
            keep.append(i)
            
        # 计算相交区域的左上坐标及右下坐标
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        # 计算相交的面积，不重叠时为0
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        
        # 计算 IOU = 重叠面积 / (面积1 + 面积2 - 重叠面积)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留 IOU 小于阈值的 bboxes
        inds = np.where(iou <= thresh)[0]
        if inds.size == 0:
            break
        order = order[inds + 1] # 因为我们上面求iou的时候得到的结果索引与order相比偏移了一位，因此这里要补回来
    return keep  # 这里返回的是bboxes中的索引，根据这个索引就可以从bboxes中得到最终的检测框结果
```

### list循环迭代版本（思路清晰，面试使用）

```python
from typing import List


class Box:
    x: int
    y: int
    w: int
    h: int
    score: float

    def __init__(self, x, y, w, h, score):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score


def iou(box1: Box, box2: Box):
    """
    计算两个box的IOU
    :param box1:
    :param box2:
    :return:
    """
    xmin = max(box1.x, box2.x)
    ymin = max(box1.y, box2.y)
    xmax = min(box1.x + box1.w, box2.x + box2.w)
    ymax = min(box1.y + box1.h, box2.y + box2.h)
    inter_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    area1 = box1.w * box1.h
    area2 = box2.w * box2.h
    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def nms(boxes: List[Box], threshold: float):
    ret = []  # 返回值
    # 1. 按照置信度排序
    boxes = sorted(boxes, key=lambda x: x.score, reverse=True)  # 降序排序
    index = [_ for _ in range(0, len(boxes))]  # 记录所有待NMS的box索引，迭代维护这个数组
    while len(index) > 0:
        # 2. 选出score最高的box
        box_cur = boxes[index[0]]
        ret.append(box_cur)
        # 3. 计算其与剩余box的IOU，过滤非极大值
        for i in range(len(index) - 1, 0, -1):  # 倒序实现对index的删除操作(一定是用index的索引删除才行)
            if iou(box_cur, boxes[index[i]]) > threshold:
                del index[i]
        del index[0]
        # 此时index只保留了剩余待nms的box
    return ret


import numpy as np
from copy import deepcopy
if __name__ == '__main__':
    boxes = np.array([[100, 100, 210, 210, 0.72],
                  [250, 250, 420, 420, 0.8],
                  [220, 220, 320, 330, 0.92],
                  [100, 100, 210, 210, 0.72],
                  [230, 240, 325, 330, 0.81],
                  [220, 230, 315, 340, 0.9]])
    tmp = []
    for box in boxes:
        a = Box(*box)
        tmp.append(deepcopy(a))
    boxes = tmp
    import matplotlib.pyplot as plt
    
    
    def plot_bbox(dets: List[Box], c='k'):
        x1 = [_.x for _ in dets]
        y1 = [_.y for _ in dets]
        x2 = [_.w for _ in dets]
        y2 = [_.h for _ in dets]
    
        plt.plot([x1, x2], [y1, y1], c)
        plt.plot([x1, x1], [y1, y2], c)
        plt.plot([x1, x2], [y2, y2], c)
        plt.plot([x2, x2], [y1, y2], c)
        plt.title(" nms")
    
    
    plt.figure(1)
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    plt.sca(ax1)
    plot_bbox(boxes, 'k')  # before nms
    
    keep = nms(boxes, threshold=0.7)
    plt.sca(ax2)
    plot_bbox(keep, 'r')  # after
    plt.show()

```



## 参考资料

[NMS算法详解（附Pytorch实现代码）](https://zhuanlan.zhihu.com/p/54709759)
[非极大值抑制（Non-Maximum Suppression，NMS）](https://www.bbsmax.com/A/A2dmV1YOze/)

​																																												By Yee
​																																											2020.05.16

# 补充

## bounding box voting

> 流程：
> 1：NMS处理之后的人脸检测框之后，得到人脸检测框nmsed
> 2：Nmsed得到之后的人脸检测框和原始的检测框取交集（也就NMS的阈值）来进行选择较多的后选框；
> 3：后选框中采用加权均值的方法，得到一个最终的框；权值为每个框的置信度。
> 4：每一个框的权重根据人脸框的置信度来计算；
> 5:  再根据评分进行过滤

**投票法**可以理解为以顶尖筛选出一流，再用一流的结果进行加权投票决策。==最终的投票nmsed的框是不参与的，参与只是被NMS滤去的框==

> 需要调整的参数： 
> box voting 的阈值。(NMS的阈值)
> 不同的输入中这个框至少出现了几次来允许它输出。
> 得分的阈值，一个目标框的得分低于这个阈值的时候，就删掉这个目标框。

Box Voting[9]想法是根据NMS**被抑制掉重合度较高的box进一步refine NMS之后的边框**。主要做法步骤是：

NMS之后的边框$Y= \{S_i, B_i\}$，需要进行投票的集合![[公式]](https://www.zhihu.com/equation?tex=B_j%5Cin+N%28B_i%29) 是NMS过程中被抑制掉的检测框（重合度大于0.5）,

权重计算公式：

![[公式]](https://www.zhihu.com/equation?tex=%5Comega+_j+%3D+max%280%2CS_j%29) ，（取最大的置信度作为该框的置信度）

最终得到的边框值：（边框加权整合）

![[公式]](https://www.zhihu.com/equation?tex=B_%7Bi%7D%5E%7B%27%7D+%3D+%5Cfrac%7B%5Csum_%7Bj%3AB_j+%5Cin+N%28B_i%29%7D+%5Comega+_j+B_j%7D%7B%5Csum_%7Bj%3AB_j+%5Cin+N%28B_i%29%7D%5Comega+_j%7D) 

### 参考资料

[bounding box voting](https://blog.csdn.net/wfei101/article/details/78220184)（流程说明，但其中的公式有误）

[跟着top学套路---目标检测篇](https://zhuanlan.zhihu.com/p/50621694)（公式介绍）

## soft-nms

针对的问题是当图片中两个相同类别的物体离得比较近的时候，一般的NMS可能会出现错误抑制的情况，这是因为NMS直接将与最大score的box的IoU有大于某个阈值的其他同类别的box的score直接置为0导致的，这就有点hard了。所以**soft NMS就是不直接将这些满足条件的box直接设置为0而是降低score值**，这样循环操作。

有两种形式，线性加权和高斯加权

![[公式]](https://www.zhihu.com/equation?tex=s_i+%3D+++%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D+s_i+%2C+%5Cquad%E3%80%80if+%5Cquad+iou%3CN+%5C%5C++s_i%281-iou%29%2C+%5Cquad%E3%80%80if+%5Cquad+iou%3E%3DN+%5Cend%7Bmatrix%7D%5Cright.)

![[公式]](https://www.zhihu.com/equation?tex=s_i%3Ds_i%5Cexp+%28%5Cfrac%7B-iou%5E%7B2%7D%7D%7B%5Csigma%7D%29)

 这么做的解释如下：

<img src="https://images2018.cnblogs.com/blog/1062917/201803/1062917-20180309102358259-409293801.jpg" alt="img" style="zoom:50%;" />

如上图：

假如还检测出了3号框，而我们的最终目标是检测出1号和2号框，并且剔除3号框，原始的nms只会检测出一个1号框并剔除2号框和3号框，而softnms算法可以对1、2、3号检测狂进行置信度排序，可以知道这三个框的置信度从大到小的顺序依次为：1-》2-》3（由于是使用了惩罚，所有可以获得这种大小关系），如果我们**再选择了合适的置信度阈值（说明是先进行nms再进行的置信度筛选）**，就可以保留1号和2号，同时剔除3号，实现我们的功能。

但是，这里也有一个问题就是置信度的阈值如何选择，作者在这里依然使用手工设置的值，依然存在很大的局限性，所以该算法依然存在改进的空间。

### 参考资料

[NMS和soft-nms算法](https://www.cnblogs.com/zf-blog/p/8532228.html)

## Weighted Boxes Fusion

下面是WBF的详细算法步骤：

1. 每个模型的每个预测框都添加到List B，并将此列表按置信度得分C降序排列
2. 建立空List L 和 F（用于融合的）
3. 循环遍历B，并在F中找到于之匹配的box（同一类别MIOU > 0.55）
4. 如果 step3 中没有找到匹配的box 就将这个框加到L和F的尾部
5. 如果 step3 中找到了匹配的box 就将这个框加到L，加入的位置是box在F中匹配框的Index. L中每个位置可能有多个框，需要根据这多个框更新对应F[index]的值。

总的思路：

1. 根据box之间的IOU值进行聚类，IOU大于0.55的都聚为一类

2. 更新聚类中心：

   box坐标按置信度conf取加权平均；conf直接取平均。

3. 所有的聚类中心做为结果

使用场景：

1. 一般是两个模型各自NMS之后，进行模型融合使用。
2. 直接将NMS换成WBF效果不好。

### 参考链接

[目标检测-提升方案-目标框加权融合-Weighted Boxes Fusion笔记及源码](https://www.pianshen.com/article/49031412197/)

## DIOU-NMS

就是将判断标准由IOU换成了DIOU