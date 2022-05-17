# Anchor-free相对于Anchor-based的优势

### anchor based

#### 优点

- 使用anchor机制产生密集的anchor box，使得网络可直接在此基础上进行目标分类及边界框坐标回归。**加入了先验框，训练稳定**。
- 密集的anchor box**可有效提高网络目标召回能力**，对于小目标检测来说提升非常明显。

#### 缺点

- anchor机制中，**需要设定的超参**：尺度(scale)和长宽比( aspect ratio) 是比较难设计的。这需要较强的先验知识。**使得这些anchor集合存在数据相关性，泛化性能较差**。
- 冗余框非常之多：一张图像内的目标毕竟是有限的，基于每个anchor设定大量anchor box**会产生大量**的easy-sample，即**完全不包含目标的背景框**。这会**造成正负样本严重不平衡问题**，也是one-stage算法难以赶超two-stage算法的原因之一。使用包括two-stage的RPN和one-stage的Focal loss
- 网络实质上是看不见anchor box的，在anchor box的基础上进行边界回归更像是一种在范围比较小时候的强行记忆。
- 基于anchor box进行目标类别分类时，IOU阈值超参设置也是一个问题

### anchor free

#### 优点

- anchor-free要比anchor-base**少2/3的参数量**，因为anchor-base一个位置要预测3个长宽不同的bbox，而free只预测一个。
- 不需要像anchor一样，**不需要设置超参数**。
- <u>**容易部署**（这是anchor-free得以推广的主要原因）</u>
  主要是解码方便：直接对heatmap使用池化，就相当于做了NMS，然后利用偏移和宽高就可以获得对应的检测框。（FCOS使用了NMS，而centernet是直接对点进行池化做NMS）
  而anchor-base需要解码每个位置，再使用NMS，还<u>需要实现求解出每个anchor的位置和大小，使得解码很麻烦。</u>
- 对于JDE的MOT算法，效果更好。（参看下图理解）
  1）对于anchor-base：不同图像块（最后特征图的不同像素点）的anchor可能负责同一个目标ID，那么就很难确定应该使用哪个anchor对应的特征，造成歧义。而anchor-free只有目标中心点对应的位置负责这个目标ID，<u>不会产生歧义</u>。
  2）anchor-base的特征图很小，比如缩小了1/8，那么这个细粒度不足，而anchor-free的一般是1/4，细粒度更好。

![img](https://ask.qcloudimg.com/http-save/yehe-7255282/dvv3y762l8.png?imageView2/2/w/1620)

#### 缺点

- 正负样本极端不平衡
- 语义模糊性（两个目标中心点重叠）
  现在这两者大多是采用Focus Loss和FPN来缓解的，但并没有真正解决。
- 检测结果不稳定，需要设计更多的方法来进行re-weight

# CornerNet

左上角和右下角的类别、offset、embeding

ExtermNet



# CenterNet

## CenterNet相比One Stage和Two Stage算法的区别

1）**CenterNet没有anchor这个概念，只负责预测物体的中心点**，所以也没有所谓的box overlap大于多少多少的算positive anchor，小于多少算negative anchor这一说，也不需要区分这个anchor是物体还是背景 - 因为每个目标只对应一个中心点，这个中心点是通过heatmap中预测出来的，所以不需要NMS再进行来筛选。

2）**CenterNet的输出分辨率的下采样因子是4**，比起其他的目标检测框架算是比较小的(Mask-Rcnn最小为16、SSD为最小为16)。之所以设置为4是因为centernet没有采用FPN结构，因此所有中心点要在一个Feature map上出，因此分辨率不能太低。



看下图可以比较直观的看出centernet的建模方法

![img](https://pic1.zhimg.com/80/v2-f0871ea7277c2b6129a0c7c7142e8bec_720w.jpg)

总体来说，**CenterNet结构十分简单，直接检测目标的中心点和大小**，是真正意义上的anchor-free。

## 网络结构

整体的网络结构如下：
**总的来看就是增加了三个head，大小都是一样的，只不过分别预测了改位置是否是物体中心点、中心点的xy偏移量、以该位置为中心点的目标的长宽。**

![img](https://upload-images.jianshu.io/upload_images/5356150-ac04a64a53df0521.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构：

1.Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS

2.DLA-34 : 37.4% COCOAP and 52 FPS

3.Hourglass-104 : 45.1% COCOAP and 1.4 FPS

每个网络内部的结构不同，但是在模型的最后输出部分都是加了三个网络构造来输出预测值，默认是80个类、2个预测的中心点坐标、2个中心点的偏置。

三种网络结构如下图所示

![img](https://pic1.zhimg.com/80/v2-dec8d87ff1a2376bf7373f2114d39ca8_720w.jpg)

在整个训练的流程中，CenterNet学习了CornerNet的方法。对于每个标签图(ground truth)中的某一类，我们要将真实关键点(true keypoint) 计算出来用于训练，**中心点的计算方式**如下

![[公式]](https://www.zhihu.com/equation?tex=p%3D%5Cleft%28%5Cfrac%7Bx_%7B1%7D%2Bx_%7B2%7D%7D%7B2%7D%2C+%5Cfrac%7By_%7B1%7D%2By_%7B2%7D%7D%7B2%7D%5Cright%29)

，对于下采样后的坐标，我们设为

![[公式]](https://www.zhihu.com/equation?tex=%5Ctilde%7Bp%7D%3D%5Cleft%5Clfloor%5Cfrac%7Bp%7D%7BR%7D%5Cright%5Crfloor)

，其中 R 是文中提到的下采样因子4。所以我们**最终计算出来的中心点是对应低分辨率的中心点**。

然后我们对图像进行标记，在下采样的[128,128]图像中将ground truth point以下采样的形式，**用一个二维高斯滤波**

![[公式]](https://www.zhihu.com/equation?tex=Y_%7Bx+y+c%7D%3D%5Cexp+%5Cleft%28-%5Cfrac%7B%5Cleft%28x-%5Ctilde%7Bp%7D_%7Bx%7D%5Cright%29%5E%7B2%7D%2B%5Cleft%28y-%5Ctilde%7Bp%7D_%7By%7D%5Cright%29%5E%7B2%7D%7D%7B2+%5Csigma_%7Bp%7D%5E%7B2%7D%7D%5Cright%29)

**来将关键点分布到特征图上。**

## 损失函数

heatmap loss的计算公式如下，**对focal loss进行了改写，α和β是超参数，用来均衡难易样本和正负样本**。**N是图像的关键点数量(正样本个数)**，用于将所有的positive focal loss标准化为1，求和符号的下标xyc表示所有heatmap上的所有坐标点(c表示目标类别，每个类别一张heatmap)，![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BY%7D_%7Bx+y+c%7D)为预测值，Yxyc为标注真实值。

相比focal loss，**负样本的loss里面多了一个$(1-Yxyc)β$ 是为了抑制$0<Yxyc<1$的负样本的loss(heatmap高斯中心点附近那些点)**

![img](https://img2020.cnblogs.com/blog/1483773/202011/1483773-20201115170413294-1595586774.png)

## 目标中心的偏置损失

因为上文中对图像进行了 R=4 的下采样，这样的**特征图重新映射到原始图像上的时候会带来精度误差，因此对于每一个中心点，额外采用了一个local offset 去补偿它**。所有类 c 的中心点共享同一个offset prediction，这个偏置值(offset)**用L1 loss**来训练：

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bo+f+f%7D%3D%5Cfrac%7B1%7D%7BN%7D+%5Csum_%7Bp%7D%5Cleft%7C%5Chat%7BO%7D_%7B%5Ctilde%7Bp%7D%7D-%5Cleft%28%5Cfrac%7Bp%7D%7BR%7D-%5Ctilde%7Bp%7D%5Cright%29%5Cright%7C)

这个偏置损失是可选的，我们不使用它也可以，只不过精度会下降一些。

## 目标大小的损失

我们假设$ (X_1^{(k)}, Y_1^{(k)}, X_2^{(k)}, Y_2^{(k)})$ 为为目标 k ，所属类别为c ，它的中心点为

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bk%7D%3D%5Cleft%28%5Cfrac%7Bx_%7B1%7D%5E%7B%28k%29%7D%2Bx_%7B2%7D%5E%7B%28k%29%7D%7D%7B2%7D%2C+%5Cfrac%7By_%7B1%7D%5E%7B%28k%29%7D%2By_%7B2%7D%5E%7B%28k%29%7D%7D%7B2%7D%5Cright%29)

我们使用关键点预测 Y^ 去预测所有的中心点。然后对每个目标 K 的size进行回归，最终回归到$S_k = (X_2^{(k)}-X_1^{(k)}, Y_2^{(k)}-Y_1^{(k)})$​ ，这个值是在训练前提前计算出来的，是<u>进行了下采样之后的长宽值</u>。（<u>回归的直接就是宽高的绝对值</u>，并不像anchor-base那种预测一个和anchor的宽高有关的相对大小）

作者采用**L1 loss** 监督w,h的回归

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bs+i+z+e%7D%3D%5Cfrac%7B1%7D%7BN%7D+%5Csum_%7Bk%3D1%7D%5E%7BN%7D%5Cleft%7C%5Chat%7BS%7D+p_%7Bk%7D-s_%7Bk%7D%5Cright%7C)

整体的损失函数为物体损失、大小损失与偏置损失的和，每个损失都有相应的权重。

![[公式]](https://www.zhihu.com/equation?tex=L_%7Bd+e+t%7D%3DL_%7Bk%7D%2B%5Clambda_%7Bs+i+z+e%7D+L_%7Bs+i+z+e%7D%2B%5Clambda_%7Bo+f+f%7D+L_%7Bo+f+f%7D)

在**论文中 size 和 off的系数分别为0.1和1 ，论文中所使用的backbone都有三个head layer，分别产生[1,80,128,128]、[1,2,128,128]、[1,2,128,128]，也就是每个坐标点产生 C+4 个数据，分别是类别以及、长宽、以及偏置。**

## 测试阶段

在预测阶段，首先针对一张图像进行下采样，随后对下采样后的图像进行预测，对于每个类在下采样的特征图中预测中心点，然后将输出图中的每个类的热点单独地提取出来。具体怎么提取呢？**就是检测当前热点的值是否比周围的八个近邻点(八方位)都大(或者等于)，然后取100个这样的点，采用的方式是一个3x3的MaxPool，类似于anchor-based检测中nms的效果**。

下图展示网络模型预测出来的中心点、中心点偏置以及该点对应目标的长宽：

![img](https://pic3.zhimg.com/80/v2-66f5dc7c29dca3d8939237bab0037d22_720w.jpg)

那最终是怎么选择的，最终是根据模型预测出来的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7BY%7D+%5Cin%5B0%2C1%5D%5E%7B%5Cfrac%7BW%7D%7BR%7D+%5Ctimes+%5Cfrac%7BH%7D%7BR%7D+%5Ctimes+C%7D) ，也就是当前中心点存在物体的概率值，代码中设置的阈值为0.3，也就是从上面选出的100个结果中调出大于该阈值的中心点作为最终的结果。

### 如何选定最终的结果？

**从heatmap中选出评分最高的那个类，如果该类的评分达到阈值，那么就认为是该类，否则就认为是背景。**

## **总结**

CenterNet的**优点如下**：

1. 设计模型的**结构比较简单**，一般人也可以轻松看明白，不仅对于two-stage，对于one-stage的目标检测算法来说该网络的模型设计也是优雅简单的。

2. 该**模型的思想不仅可以用于目标检测，还可以用于3D检测和人体姿态识别**，虽然论文中没有是深入探讨这个，但是可以说明这个网络的设计还是很好的，我们可以借助这个框架去做一些其他的任务。

3. 虽然目前尚未尝试轻量级的模型，但是可以猜到这个模型**对于嵌入式端这种算力比较小的平台还是很有优势的**。

当然说了一堆优点，CenterNet的**缺点**也是有的，那就是：

1. 在实际训练中，如果在图像中，同一个类别中的某些物体的GT中心点，在下采样时会挤到一块，也就**是两个物体在GT中的中心点重叠了，CenterNet对于这种情况也是无能为力的，也就是将这两个物体的当成一个物体来训练(因为只有一个中心点)**。同理，在预测过程中，如果两个同类的物体在下采样后的中心点也重叠了，那么CenterNet也是只能检测出一个中心点，不过CenterNet对于这种情况的处理要比faster-rcnn强一些的，具体指标可以查看论文相关部分。

2. 有一个需要注意的点，CenterNet在训练过程中，**如果同一个类的不同物体的高斯分布点互相有重叠，那么则在重叠的范围内选取较大的高斯点**。

## 参考链接

[真Anchor Free目标检测----CenterNet详解](https://zhuanlan.zhihu.com/p/72373052)——主要参考这个，部分图参考的其他的

# FCOS

![FCOS:一阶全卷积目标检测](https://pica.zhimg.com/v2-309265c2ea3fea05f958c07203780be8_1440w.jpg?source=172ae18b)

中心点类别、offset、**边框距中心点的距离**

### **1.全卷积一阶检测器**

FCOS首先使用Backone CNN(用于提取特征的主干架构CNN)，另*s*为feature map之前的总步伐。

***与anchor-based检测器的区别***

***第一点***

- anchor-based算法将输入图像上的位置作为锚框的中心店，并且对这些锚框进行回归。
- **FCOS直接对feature map中每个位置对应原图的边框都进行回归**，换句话说FCOS直接把每个位置都作为训练样本，这一点和FCN用于语义分割相同。
  这一点和CenterNet也不同

> FCOS算法feature map中位置与原图对应的关系，如果feature map中位置为![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) ,映射到输入图像的位置是 ![[公式]](https://www.zhihu.com/equation?tex=%28%5Clfloor+%5Cfrac%7Bs%7D%7B2%7D+%5Crfloor%2Bxs%2C%5Clfloor+%5Cfrac%7Bs%7D%7B2%7D+%5Crfloor%2Bys%29) 。

***第二点***

- 在训练过程中，anchor-based算法对样本的标记方法是，如果anchor对应的边框与真实边框(ground truth)交并比大于一定阈值，就设为正样本，并且把交并比最大的类别作为这个位置的类别。
- 在FCOS中，如果位置 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 落入**任何**真实边框，就认为它是一个正样本，它的类别标记为这个真实边框的类别。

> 这样会带来一个问题，如果标注的真实边框重叠，位置 ![[公式]](https://www.zhihu.com/equation?tex=%28x%2Cy%29) 映射到原图中落到多个真实边框，这个位置被认为是模糊样本，后面会讲到用**多级预测**的方式解决的方式解决模糊样本的问题。

***第三点***

- 以往算法都是训练一个多元分类器
- FCOS训练 ![[公式]](https://www.zhihu.com/equation?tex=C) 个二元分类器(*C*是类别的数目)——同YOLOv4的多个

***与anchor-based检测器相似之处***

与anchor-based算法的相似之处是FCOS算法训练的目标同样包括两个部分：位置和类别。

FCOS算法的损失函数为：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+L%5Cleft%28%5Cleft%5C%7B%5Cboldsymbol%7Bp%7D_%7Bx%2C+y%7D%5Cright%5C%7D%2C%5Cleft%5C%7B%5Cboldsymbol%7Bt%7D_%7Bx%2C+y%7D%5Cright%5C%7D%5Cright%29+%26%3D%5Cfrac%7B1%7D%7BN_%7B%5Ctext+%7B+pos+%7D%7D%7D+%5Csum_%7Bx%2C+y%7D+L_%7B%5Ctext+%7B+cls+%7D%7D%5Cleft%28%5Cboldsymbol%7Bp%7D_%7Bx%2C+y%7D%2C+c_%7Bx%2C+y%7D%5E%7B%2A%7D%5Cright%29+%5C%5C+%26%2B%5Cfrac%7B%5Clambda%7D%7BN_%7B%5Ctext+%7B+pos+%7D%7D%7D+%5Csum_%7Bx%2C+y%7D+%5Cmathbb%7B1%7D_%7B%5Cleft%5C%7Bc_%7Bx%2C+y%7D%3E0%5Cright%5C%7D%7D+L_%7B%5Coperatorname%7Breg%7D%7D%5Cleft%28%5Cboldsymbol%7Bt%7D_%7Bx%2C+y%7D%2C+%5Cboldsymbol%7Bt%7D_%7Bx%2C+y%7D%5E%7B%2A%7D%5Cright%29+%5Cend%7Baligned%7D)

其中 ![[公式]](https://www.zhihu.com/equation?tex=L_%7Bcls%7D) 是类别损失， ![[公式]](https://www.zhihu.com/equation?tex=L_%7Breg%7D) 是交并比的损失。

### **3.Center-ness**

<img src="https://pic2.zhimg.com/80/v2-6300c9570dcb7196aa07a012443345bd_720w.jpg" alt="img" style="zoom:50%;" />

<u>由于FCOS将GT涵盖的所有位置都作为正样本回归，这造成了**距离目标中心较远的位置**产生很多低质量的预测边框。</u>

在FCOS中提出了一种简单而有效的策略来<u>抑制这些低质量的预测边界框</u>，而且不引入任何超参数。具体来说，FCOS添加单层分支，与分类分支并行，以预测"Center-ness"位置。

![[公式]](https://www.zhihu.com/equation?tex=cenerness%5E%7B%2A%7D%3D%5Csqrt%7B%5Cfrac%7B%5Cmin+%5Cleft%28l%5E%7B%2A%7D%2C+r%5E%7B%2A%7D%5Cright%29%7D%7B%5Cmax+%5Cleft%28l%5E%7B%2A%7D%2C+r%5E%7B%2A%7D%5Cright%29%7D+%5Ctimes+%5Cfrac%7B%5Cmin+%5Cleft%28t%5E%7B%2A%7D%2C+b%5E%7B%2A%7D%5Cright%29%7D%7B%5Cmax+%5Cleft%28t%5E%7B%2A%7D%2C+b%5E%7B%2A%7D%5Cright%29%7D%7D)

center-ness(可以理解为一种具有度量作用的概念，在这里称之为**"中心度"**)，中心度取值为0,1之间，使用交叉熵损失进行训练。并把损失加入前面提到的损失函数中。测试时，**将预测的中心度与相应的分类分数相乘，计算最终得分**(用于对检测到的边界框进行排序)。因此，**中心度可以降低远离对象中心的边界框的权重**。因此，这些低质量边界框很可能被最终的非最大抑制（NMS）过程滤除，从而显着提高了检测性能。

**总结：Center-ness将重点关注位于目标中心的特征点，而不是靠近边缘的点。**——有点类似于YOLO中的IOU分支，用于判断该处还有目标的概率

### 正负样本选择

1. 确定GT应该由哪个特征层负责。
   <u>对于第一个特征图上的点，如果该点落在了GT内，而且满足**0 < max(中心点到4条边的距离) < 64**，那么该gt bbox就属于第1层负责</u>，其余层也是采用类似原则。总结来说就是第1层负责预测尺度在`0~64`范围内的gt，第2层负责预测尺度在`64~128`范围内的gt，其余类推。通过该分配策略就可以将不同大小的gt分配到最合适的预测层进行学习。
2. 落在GT范围内的所有点都是正样本。
   如果<u>一个点落入多个GT内</u>，那么就让该点<u>负责较小面积的GT</u>。

<img src="https://pic1.zhimg.com/80/v2-55de485498ea184e602d8e2a1dd041d4_720w.jpg" alt="img" style="zoom:50%;" />

### 参考链接

[FCOS:一阶全卷积目标检测](https://zhuanlan.zhihu.com/p/63868458)

[目标检测正负样本区分策略和平衡策略总结(二)](https://zhuanlan.zhihu.com/p/138828372)——[FCOS难点](https://zhuanlan.zhihu.com/p/137860293)——关于正负样本选择

