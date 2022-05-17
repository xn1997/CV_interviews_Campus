## 问题

在网络模型当中，经常要进行不同通道特征图的信息融合相加操作，以整合不同通道的信息，在具体实现方面特征的融合方式一共有两种，一种是 ResNet 和 FPN 等当中采用的 element-wise add ，另一种是 DenseNet 等中采用的 concat 。他们之间有什么区别呢？

## add

以下是 keras 中对 add 的实现源码：

```python
def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
        output += inputs[i]
    return output
```

其中 inputs 为待融合的特征图，inputs[0]、inputs[1]……等的通道数一样，且特征图宽与高也一样。

从代码中可以很容易地看出，**add 方式有以下特点**：

1. 做的是对应通道对应位置的值的相加，通道数不变
2. 描述图像的特征个数不变，但是每个特征下的信息却增加了。

## concat

阅读下面代码实例帮助理解 concat 的工作原理：

```python
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# tensor t3 with shape [2, 3]
# tensor t4 with shape [2, 3]
tf.shape(tf.concat([t3, t4], 0)) ==> [4, 3]
tf.shape(tf.concat([t3, t4], 1)) ==> [2, 6]
--------------------- 
作者：KUNLI7 
来源：CSDN 
原文：https://blog.csdn.net/u012193416/article/details/79479935 
版权声明：本文为博主原创文章，转载请附上博文链接！
```

在模型网路当中，数据通常为 4 个维度，即 num×channels×height×width ，因此默认值 1 表示的是 channels 通道进行拼接。如：

```python
combine = torch.cat([d1, add1, add2, add3, add4], 1)
```

从代码中可以很容易地看出，concat 方式有以下特点：

1. 做的是通道的合并，通道数变多了
2. 描述图像的特征个数变多，但是每个特征下的信息却不变。

## 多一点理解

>add相当于加了一种prior，当两路输入可以具有“对应通道的特征图语义类似”的性质的时候，可以用add来替代concat，这样更节省参数和计算量（concat是add的2倍）

1. add默认两个特征图**对应的通道所要表达的信息类似**，类似的信息也就自然可以相加到一起。

   1. 比如对应通道的尺度一致（信息类似，尺度理论上也一致），那么相加就相当于是融合了多个通道的信息（类似于reid中多模型相似度矩阵的ensemble）；

   2. 而尺度不一致时，直接相加，那么尺度小的就会被尺度大的特征所淹没，此时就不应该使用add。

   然而如何判断信息是否类似没办法得知，属于一个黑盒子

2. concat直接通过训练学习来整合两个特征图通道之间的信息，相比add，更能提取出合适的信息，效果更好。

3. **总之：**

   add：优点：计算量少。缺点：特征提取能力差。

   concat：优点：特征提取能力强。缺点：计算量大。

## 参考资料

[理解concat和add的不同作用](https://blog.csdn.net/qq_32256033/article/details/89516738)
[卷积神经网络中的add和concatnate区别](https://blog.csdn.net/weixin_39610043/article/details/87103358)

