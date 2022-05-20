# YOLOv1

![img](https://pic4.zhimg.com/80/v2-563f60701e6572b530b7675eabd0cf47_720w.jpg)

#### 正负样本选取

如果目标的中心落在cell中，那么这个cell就负责预测这个类别。
由于每个cell预测两个bbox，那么选择与GT IOU大的bbox来预测这个目标，也就是这一个框的$1_{ij}^{obj}=1,1_{ij}^{nobj}=0$​，其余不负责目标的都视为没有目标，即$1_{ij}^{obj}=0,1_{ij}^{nobj}=1$​​​。
如果多个目标的中心落入同一个cell，那么仍然**选取IOU最大**的一组GT和bbox进行预测。

结论：
可以看出，整个图像中有几个GT，那么就有几个正样本，样本严重不平衡。

#### 前向过程

1. 最后一个feature map一定为7*7
2. 每个feature map cell（特征图中所有channel中的某一个像素）预测两个Bbox，一个是否有目标和20个目标的评分（**仍然使用FC不是全卷积**）
3. 特点：直接预测边框和评分，类似RPN（YOLO只预测两个BBox，RPN预测9个anchor）；没有寻找proposal阶段，速度快但精度低
4. 简易结构：CNN+分类和回归

feature map每个单元（像素）输出：2*（4+1）+20。

2：2个bounding box（类似anchor，用于目标定位）（并不是anchor，anchor是不同长宽比尺度的预选框，而这里仅仅是为了回归两个结果，没有anchor中长宽比的概念）；

4：box的4个值；

1：该box的置信度=**该box有目标概率（1或0）× 与真实box的IOU**。<img src="https://pic1.zhimg.com/80/v2-fa1bd4707f44d9c542aa4e29267f3978_720w.jpg" style="zoom:30%;" />

20：该cell的类别，注意两个box都是这一个类别。

- 在test的时候，**每个网格预测的class信息和bounding box预测的confidence信息相乘**，就得到每个bounding box的class-specific confidence score:

  <img src="https://pic2.zhimg.com/80/v2-80ac96115524cf3112a33de739623ac5_720w.png" alt="img" style="zoom:30%;" />

  等式左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息。

- 得到每个box的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行NMS处理，就得到最终的检测结果。

#### loss

<img src="https://raw.githubusercontent.com/xn1997/picgo/master/v2-aad10d0978fe7bc62704a767eabd0b54_720w.jpg" style="zoom:80%;" />

- 给予坐标损失更大的loss weight。
- 对没有object的box的confidence loss，赋予小的loss weight。
- 有object的box的confidence loss和类别的loss的loss weight正常取1。
- 只有当某个网格中有object的时候才对classification error进行惩罚。
- 只有当某个box  predictor对某个ground truth box负责的时候，才会对box的coordinate error进行惩罚。而对哪个ground truth box负责就看其预测值和ground truth box的IoU是不是在那个cell的所有box中最大

![c65a416e27972fef0cfd6d16bf53fea](https://raw.githubusercontent.com/xn1997/picgo/master/c65a416e27972fef0cfd6d16bf53fea.jpg)

#### 缺点

- 由于输出层为全连接层，因此在检测时，YOLO训练模型只支持与训练图像相同的输入分辨率。
- <font color='red'>无法解决拥挤问题，</font>虽然每个格子可以预测B个bounding box，但是最终只选择只选择IOU最高的bounding  box作为物体检测输出，即每个格子最多只预测出一个物体。
- YOLO loss函数中，大物体IOU误差和小物体IOU误差对网络训练中loss贡献值接近（虽然采用求平方根方式，使大目标的产生的损失减小，小目标的损失相应增加，但没有根本解决问题）。因此，**对于小物体，小的IOU误差也会对网络优化过程造成很大的影响，从而降低了物体检测的定位准确性**。

#### 与Fast-RCNN对比优缺点：

1. Fast RCNN的回归过程只使用了ROI内的图像，无法看到全图信息，所以区分背景的能力差，背景的误差大。
   YOLO可以看到全图，区分背景和物体的能力强，背景的误差小。
2. Fast RCNN的定位精度高。
   YOLO的定位精度差。

# YOLOv2

输出：5*（4+1）+ ？变成了5个anchor，结合coco（检测数据集20类）和ImageNet（分类数据集）使得分类总数达到9000

文章提出了一种新的**训练方法–联合训练算法**，这种算法可以把这两种的数据集混合到一起。使用一种分层的观点对物体进行分类，用巨量的**分类数据集数据来扩充检测数据集**，从而把两种不同的数据集混合起来。

联合训练算法的基本思路就是：同时在检测数据集和分类数据集上训练物体检测器（Object Detectors ），**用检测数据集的数据学习物体的准确位置，用分类数据集的数据来增加分类的类别量、提升健壮性。**

#### 改进

##### BN层

##### 借鉴RPN使用anchor base

- 为什么下采样32倍为后为奇数，而不是偶数？

  由于**图片中的物体都倾向于出现在图片的中心位置**，特别是那种比较大的物体，所以有一个单独位于物体中心的位置用于预测这些物体。YOLO的卷积层采用32这个值来下采样图片，所以通过选择416×416用作输入尺寸最终能输出一个13×13的Feature Map。使用Anchor Box会让**精度稍微下降**，但用了它能让YOLO能预测出大于一千个框，同时recall从81%达到88%，mAP达到69.2%

**召回率升高，mAP轻微下降的原因是：**因为YOLOV2不使用anchor  boxes时，每个图像仅预测98个边界框。但是使用anchor  boxes，YOLOV2模型预测了一千多个框，由于存在很多无用的框，**导致训练受到大量负样本的影响使检测框回归困难，mAP值下降**。但是由于预测的框多了，所以**能够预测出来的属于ground truth的框就多了**，所以召回率就增加了。目标检测不是只以mAP为指标的，有些应用场景下要求召回率高。

##### 聚类提取先验框的尺度信息

如果我们用标准的欧式距离的k-means，尺寸大的框比小框产生更多的错误。因为我们的目的是提高IOU分数，这依赖于Box的大小，所以距离度量使用IOU距离：

![](https://pic4.zhimg.com/80/v2-188fa2572453119bbe96dd00102b7163_720w.png)

centroid是聚类时被选作中心的边框，box就是其它边框，d就是两者间的“距离”。IOU越大，“距离”越近。

##### 约束预测边框的位置

由于 ![[公式]](https://www.zhihu.com/equation?tex=t_x%2Ct_y) 的取值没有任何约束，因此预测边框的中心可能出现在任何位置，训练早期阶段不容易稳定。YOLO调整了预测公式，==将预测边框的中心约束在特定gird网格内==。 

![](https://www.zhihu.com/equation?tex=%5C%5C+b_x%3D%CF%83%28t_x%29%2Bc_x++%5C%5C+b_y%3D%CF%83%28t_y%29%2Bc_y++%5C%5C+b_w%3Dp_we%5E%7Bt_w%7D++%5C%5C+b_h%3Dp_he%5E%7Bt_h%7D++%5C%5C+Pr%28object%29%E2%88%97IOU%28b%2Cobject%29%3D%CF%83%28t_o%29+)

其中， ![[公式]](https://www.zhihu.com/equation?tex=t_x%2Ct_y%2Ct_w%2Ct_h%2Ct_o)是要学习的参数，分别用于预测边框的中心和宽高，以及置信度。 σ是sigmoid函数。  ![[公式]](https://www.zhihu.com/equation?tex=c_x%2Cc_y)**是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1。  ![[公式]](https://www.zhihu.com/equation?tex=p_w%2Cp_h) 是先验框的宽和高**。 ![[公式]](https://www.zhihu.com/equation?tex=b_x%2Cb_y%2Cb_w%2Cb_h) 是预测边框的中心和宽高。 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%E2%88%97IOU%28b%2Cobject%29) 是预测边框的置信度，YOLO1是直接预测置信度的值，这里对预测参数 ![[公式]](https://www.zhihu.com/equation?tex=t_o) 进行σ变换后作为置信度的值。

##### 多尺度图像训练

每10个Batches，网络会随机地选择一个新的图片尺寸，由于使用了下采样参数是32，所以不同的尺寸大小也选择为32的倍数{320，352…..608}，最小$320*320$，最大$608*608$

##### 分层分类(YOLO9000)

通过将两个数据集混合训练，**如果遇到来自分类集的图片则只计算分类的Loss，遇到来自检测集的图片则计算完整的Loss。**

- 与RPN相比为什么精度会低
  - RPN最后的anchor选取方法也是在最后的feature map中滑动选取
  - 但RPN最后一层并没有像YOLO一样直接限制在了7*7，而是和原图相比尺寸并没有差很多，所以最后提取出的anchor数量就会非常多，而且相对来说框更小，可以检测更加小的目标
  - 可以说YOLO和RPN很相似

# YOLOv3

![](https://pic3.zhimg.com/80/v2-5d97a1b944276ee2790febd230bb2112_720w.jpg)

#### 改进

- 多尺度预测 （引入FPN）。
- 更好的基础分类网络（darknet-53, 类似于ResNet引入残差结构）。
- 分类器不在使用Softmax，分类损失采用binary cross-entropy loss（二分类交叉损失熵）

YOLOv3不使用Softmax对每个框进行分类，主要考虑因素有两个：

1. Softmax使得每个框分配一个类别（score最大的一个），而对于Open Images这种数据集，目标可能有重叠的类别标签，因此Softmax不适用于多标签分类。
2. Softmax可被独立的多个logistic分类器替代，且准确率不会下降。

分类损失采用binary cross-entropy loss。

##### 提出Darknet-53

- 将卷积层的stride设为2来达到下采样的效果，不使用池化层。
- 借鉴ResNet结构，使用shortcut连接。
- 借鉴FPN结构，使用多尺度预测。
- 预测支路采用全卷积的结构。其中最后一个卷积层的卷积核个数是255，是针对COCO数据集的80类：3*(80+4+1)=255，3表示一个grid cell包含3个bounding box，4表示框的4个坐标信息，1表示objectness score。

<img src="https://pic4.zhimg.com/80/v2-ffee273451c8bfa23124f6aa4f314413_720w.jpg" style="zoom:50%;" />

##### YOLOV3中的边框回归（同YOLO2，这里介绍的更加详细）

yolov3是在训练的数据集上**聚类**产生prior  boxes的一系列宽高(是在图像416x416的坐标系里)，默认9种。YOLOV3思想理论是将输入图像分成SxS个格子（有三处进行检测，分别是在52x52, 26x26, 13x13的feature map上，即S会分别为52,26,13），若某个物体Ground  truth的中心位置的坐标落入到某个格子，那么这个格子就负责检测中心落在该栅格中的物体。**三次检测，**每次对应的感受野不同，**32倍降采样的感受野最大（13x13），适合检测大的目标**，每个cell的三个anchor boxes为(116 ,90),(156 ,198)，(373 ,326)。**16倍（26x26）适合一般大小的物体**，anchor boxes为(30,61)， (62,45)，(59,119)。8**倍的感受野最小（52x52），适合检测小目标**，因此anchor boxes为(10,13)，(16,30)，(33,23)。所以当输入为416×416时，实际总共有(52×52+26×26+13×13)×3=10647个proposal boxes。

<img src="https://pic2.zhimg.com/80/v2-9e8c062ccb787cbfc4cc5e00fcb84c39_720w.jpg" alt="img" style="zoom:50%;" />

图 2：带有维度先验和定位预测的边界框。我们边界框的宽和高以作为离聚类中心的位移，并使用 Sigmoid 函数预测边界框相对于滤波器应用位置的中心坐标。

yolov3对每个bounding box预测偏移量和尺度缩放四个值 ![[公式]](https://www.zhihu.com/equation?tex=t_%7Bx%7D%2Ct_%7By%7D%2Ct_%7Bw%7D%2Ct_%7Bh%7D) （网络需要学习的目标），对于预测的cell（一幅图划分成S×S个网格cell）根据图像左上角的偏移 ![[公式]](https://www.zhihu.com/equation?tex=%28c_%7Bx%7D%2Cc_%7By%7D%29) ，**每个grid cell在feature map中的宽和高均为1**，以及预设的anchor box的宽和高 ![[公式]](https://www.zhihu.com/equation?tex=p_%7Bw%7D%2Cp_%7Bh%7D) （预设聚类的宽高需要除以stride映射到feature map上）。最终得到的边框坐标值是$b*$,而网络学习目标是$t*$，用sigmod函数、指数转换。可以对bounding box按如下的方式进行预测：

<img src="https://pic2.zhimg.com/80/v2-aca06a1b0594cff33b038802d7ce2295_720w.jpg" alt="img" style="zoom:70%;" />

**公式中为何使用sigmoid函数呢？**

YOLO不预测边界框中心的绝对坐标，它预测的是偏移量，预测的结果通过一个sigmoid函数，迫使输出的值在0~1之间。例如，若对中心的预测是(0.4,0.7)，左上角坐标是(6,6)，那么中心位于13×13特征地图上的(6.4,6.7)。若预测的x，y坐标大于1，比如(1.2,0.7)，则中心位于(7.2,6.7)。注意现在中心位于图像的第7排第8列单元格，这打破了YOLO背后的理论，因为如果假设原区域负责预测某个目标，目标的中心必须位于这个区域中，而不是位于此区域旁边的其他网格里。为解决这个问题，输出是通过一个sigmoid函数传递的，该函数在0到1的范围内缩放输出，有效地将中心保持在预测的网格中。

==其实图像在输入之前是按照图像的长边缩放为416，短边根据比例缩放(图像不会变形扭曲)，然后再对短边的两侧填充至416，这样就保证了输入图像是416*416的。==

- 改进：
  - 对小物体的检测率提高
    - 借鉴了多尺度检测：类似SSD，但有所区别，SSD是直接使用low-layer预测，而YOLOv3是将low-layer+top-layer上采样进行预测，准确率更高些

  - 分类网络使用Darknet-53

    - Darknet-53借鉴了Resnet的shortcut connection，准确度较高，但速度更快

  - 可以实现一个目标的多个label检测

    - open image数据集中，一个object可以有两个标签（如person and woman）
    - 使用logistic替代softmax完成多label object detection

    - - 方法：？？？

##### 多尺度预测：更好地对应不同大小的目标物体

每种尺度预测3个box, anchor的设计方式仍然使用聚类,**得到9个聚类中心,将其按照大小均分给3个尺度**.

- 尺度1: 在基础网络之后添加一些卷积层再输出box信息.
-  尺度2: 从尺度1中的倒数第二层的卷积层上采样(x2)再与最后一个16x16大小的特征图相加,再次通过多个卷积后输出box信息.相比尺度1变大两倍.
-  尺度3: 与尺度2类似,使用了32x32大小的特征图.

网络必须具备能够“看到”不同大小的物体的能力。并且网络越深，特征图就会越小，所以越往后小的物体也就越难检测出来。SSD中的做法是，在不同深度的feature map获得后，直接进行目标检测，这样小的物体会在相对较大的feature map中被检测出来，而大的物体会在相对较小的feature  map被检测出来，从而达到对应不同scale的物体的目的。然而在实际的feature map中，深度不同所对应的feature map包含的信息就不是绝对相同的。只利用浅层高分辨率的特征图进行预测将会导致特征信息不足，因此需要使用FPN结构将高层信息和底层信息结合。

<img src="https://pic1.zhimg.com/80/v2-ab6d008a835961ea29d1143a2951584c_720w.jpg" alt="img" style="zoom:70%;" />

如上图所示，对于多重scale，目前主要有以下几种主流方法。

**(a) Featurized image pyramid:** 这种方法最直观。首先对于一幅图像建立图像金字塔，不同级别的金字塔图像被输入到对应的网络当中，用于不同scale物体的检测。但这样做的结果就是每个级别的金字塔都需要进行一次处理，速度很慢。

**(b) Single feature map:** 检测只在最后一个feature map阶段进行，这个结构无法检测不同大小的物体。

**(c)  Pyramidal feature hierarchy:** 对不同深度的feature map分别进行目标检测。SSD中采用的便是这样的结构。每一个feature map获得的信息仅来源于之前的层，之后的层的特征信息无法获取并加以利用。

**(d) Feature Pyramid Network** 与(c)很接近，但有一点不同的是，当前层的feature map会对未来层的feature map进行上采样，并加以利用。这是一个有跨越性的设计。因为有了这样一个结构，当前的feature  map就可以获得“未来”层的信息，这样的话低阶特征与高阶特征就有机融合起来了，提升检测精度。

##### ResNet残差结构：更好地获取物体特征

YOLOv3中使用了ResNet结构（对应着在上面的YOLOv3结构图中的Residual Block）。Residual Block是有一系列卷基层和一条shortcut path组成。shortcut如下图所示。

![img](https://pic4.zhimg.com/80/v2-9948465532b9f5891c3d2b0fa8f4840b_720w.jpg)Residual Block

图中曲线箭头代表的便是shortcut path。除此之外，此结构与普通的CNN结构并无区别。随着网络越来越深，学习特征的难度也就越来越大。但是如果我们加一条shortcut  path的话，学习过程就从直接学习特征，变成在之前学习的特征的基础上添加某些特征，来获得更好的特征。这样一来，一个复杂的特征H(x)，之前是独立一层一层学习的，现在就变成了这样一个模型：H(x)=F(x)+x，其中x是shortcut开始时的特征，而F(x)就是对x进行的填补与增加，成为残差。因此**学习的目标就从学习完整的信息，变成学习残差了。这样以来学习优质特征的难度就大大减小了**。

##### 替换softmax层：对应多重label分类

Softmax层被替换为一个1x1的卷积层+logistic激活函数的结构。使用softmax层的时候其实已经假设每个输出仅对应某一个单个的class，但是在某些class存在重叠情况（例如woman和person）的数据集中，使用softmax就不能使网络对数据进行很好的拟合。

##### loss

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200622211901413.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2JibGluZ2JibGluZw==,size_16,color_FFFFFF,t_70)

**回归loss会乘以一个(2-wxh)的比例系数，用来加大对小box的损失。**

# YOLOv3-tiny

<img src="https://pic2.zhimg.com/80/v2-45602f74f6e0787ea9c1495a78f8ab85_720w.jpg" alt="img" style="zoom:50%;" />

TinyYOLOv3只用了两个尺度，stride=32和stride=16，这里需要注意一点的是512层和1024层之间的maxpooling的stride=1，而不是2，因此，为了保证输入输出不改变feature map的宽高，需要补左0右1、上0下1的zero padding，用PyTorch来写就是：

```python
nn.ZeroPad2d((0,1,0,1))
```

# YOLOv4

具体参考《110_YOLOv4详解.md》

## 目的及主要贡献

Yolo-V4的**主要目的在于设计一个能够应用于实际工作环境中的快速目标检测系统，且能够被并行优化，并没有很刻意的去追求理论上的低计算量**（BFLOP）。同时，Yolo-V4的作者希望算法能够很轻易的被训练，也就是说拥有一块常规了GTX-2080ti或者Titan-XP GPU就能够训练Yolo-V4, 同时能够得到一个较好的结果。整个introduction可以总结为以下几点：

- 研究设计了一个简单且高效的目标检测算法，该算法降低了训练门槛，使得普通人员在拥有一块1080TI或者2080TI的情况下就能够训练一个super fast and accurate 的目标检测器
- 在训练过程中，验证了最新的Bag-of-Freebies和Bag-of-Specials对Yolo-V4的影响
- 简化以及优化了一些最新提出的算法，包括（CBN，PAN，SAM），从而使Yolo-V4能够在一块GPU上就可以训练起来。

==主要贡献==：

1. 提出了一种高效而强大的目标检测模型。它使每个人都可以使用1080 Ti或2080 Ti GPU 训练超快速和准确的目标检测器（牛逼！）。

2. 在检测器训练期间，验证了SOTA的Bag-of Freebies 和Bag-of-Specials方法的影响。

3. 改进了SOTA的方法，使它们更有效，更适合单GPU训练，包括CBN [89]，PAN [49]，SAM [85]等。文章将目前主流的目标检测器框架进行拆分：input、backbone、neck 和 head.

具体如下图所示：

<img src="https://pic4.zhimg.com/80/v2-3f65c8ef82fe91d891fb1f9924f8c32f_720w.jpg" alt="img" style="zoom:70%;" />

- 对于GPU，作者在卷积层中使用：CSPResNeXt50 / CSPDarknet53  
- 对于VPU，作者使用分组卷积，但避免使用（SE）块-具体来说，它包括以下模型：EfficientNet-lite / MixNet / GhostNet / MobileNetV3

作者的目标是在输入网络分辨率，卷积层数，参数数量和层输出（filters）的数量之间找到最佳平衡。

## 网络结构

- **Backbone：CSPDarknet53**
- **Neck：SPP，PAN**
- **Head：YOLOv3**

**YOLOv4 =** **CSPDarknet53+SPP+PAN+YOLOv3**

其中YOLOv4用到相当多的技巧：

- **用于backbone的BoF：CutMix和Mosaic数据增强，DropBlock正则化，Class label smoothing**
- **用于backbone的BoS：Mish激活函数，CSP，MiWRC**
- **用于检测器的BoF：CIoU-loss，CmBN，DropBlock正则化，Mosaic数据增强，Self-Adversarial 训练，消除网格敏感性，对单个ground-truth使用多个anchor，Cosine annealing  scheduler，最佳超参数，Random training shapes**
- **用于检测器的Bos：Mish激活函数，SPP，SAM，PAN，DIoU-NMS**

# 参考

[目标检测之YOLO算法：YOLOv1,YOLOv2,YOLOv3,TinyYOLO，YOLOv4,YOLOv5,YOLObile,YOLOF详解](https://zhuanlan.zhihu.com/p/136382095)