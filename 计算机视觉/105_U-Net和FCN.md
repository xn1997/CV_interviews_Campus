## FCN

**相比于普通分类网络而言：FCN把后面几个全连接都换成卷积，这样就可以获得一张2维的feature map，后接softmax获得每个像素点的分类信息，从而解决了像素级分割问题。**

整个FCN网络基本原理如图5**（只是原理示意图）**：

1. image经过多个conv和+一个max pooling变为pool1 feature，宽高变为1/2
2. pool1 feature再经过多个conv+一个max pooling变为pool2 feature，宽高变为1/4
3. pool2 feature再经过多个conv+一个max pooling变为pool3 feature，宽高变为1/8
4. ......
5. 直到pool5 feature，宽高变为1/32。

![img](https://pic1.zhimg.com/80/v2-721ef7417b32a5aa4973f1e8dd16d90c_720w.jpg)图5 FCN网络结构示意图

那么：

1. 对于FCN-32s，直接对pool5 feature进行32倍上采样获得32x upsampled feature，再对32x upsampled  feature每个点做softmax prediction获得32x upsampled feature prediction（即分割图）。
2. 对于FCN-16s，首先对pool5 feature进行2倍上采样获得2x upsampled feature，再把pool4 feature和2x upsampled feature**逐点相加**，然后对相加的feature进行16倍上采样，并softmax prediction，获得16x upsampled feature prediction。
3. 对于FCN-8s，首先进行pool4+2x upsampled feature**逐点相加**，然后又进行pool3+2x upsampled**逐点相加**，即进行更多次特征融合。具体过程与16s类似，不再赘述。

作者在原文种给出3种网络结果对比，明显可以看出效果：FCN-32s < FCN-16s < FCN-8s，即**使用多层feature融合有利于提高分割准确性**。

![img](https://pic4.zhimg.com/80/v2-8c212e15670c9accca37c57c90f3df7f_720w.jpg)

## U-Net

![](https://pic4.zhimg.com/80/v2-728cbbfbb540426ad3c3fafed17c485b_720w.jpg)

==与FCN相比，仅仅是特征融合方式变成了concat而不是直接add==

## 总结

**CNN图像语义分割也就基本上是这个套路：**

1. **下采样+上采样：Convlution + Deconvlution／Resize**
2. **多尺度特征融合：特征逐点相加／特征channel维度拼接**
3. **获得像素级别的segement map：对每一个像素点进行判断类别**

# 参考

[图像语义分割入门+FCN/U-Net网络解析](https://zhuanlan.zhihu.com/p/31428783)