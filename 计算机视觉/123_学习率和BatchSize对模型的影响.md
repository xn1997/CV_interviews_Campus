随机梯度下降算法的原理如下，

<img src="https://pic4.zhimg.com/80/v2-ad229aeae91bdc7a73b54052fa264cbf_720w.jpg" alt="img" style="zoom: 25%;" />

n是批量大小(batchsize)，η是学习率(learning rate)。可知道除了梯度本身，**这两个因子直接决定了模型的权重更新**，从优化本身来看它们是影响模型性能收敛最重要的参数。

**学习率直接影响模型的收敛状态，batchsize则影响模型的泛化性能**，两者又是分子分母的直接关系，相互也可影响

## 学习率如何调整

### 初始学习率大小对模型性能的影响

初始的学习率肯定是有一个最优值的，**过大则导致模型不收敛，过小则导致模型收敛特别慢或者无法学习**，下图展示了不同大小的学习率下模型收敛情况的可能性，图来自于cs231n。



<img src="https://pic1.zhimg.com/80/v2-d9640b0170f3371f97d16683f0a39594_720w.jpg" alt="img" style="zoom:50%;" />

通常可以采用最简单的**<big>搜索法</big>，即从小到大开始训练模型，然后记录损失的变化**，通常会记录到这样的曲线。

![img](https://pic3.zhimg.com/80/v2-0ec63261a52e342113964e853708e7ca_720w.jpg)

**随着学习率的增加，损失会慢慢变小，而后增加，而最佳的学习率就可以从其中损失最小的区域选择**。

有经验的工程人员常常根据自己的经验进行选择，比如0.1，0.01等。

**随着学习率的增加，模型也可能会从欠拟合过度到过拟合状态**，在大型数据集上的表现尤其明显，笔者之前在Place365上使用DPN92层的模型进行过实验。随着学习率的增强，模型的训练精度增加，直到超过验证集。

![img](https://pic1.zhimg.com/80/v2-3d4d656ce650ce528fe00ac3e80693ec_720w.jpg)

### 不同的学习策略

#### **Mutistep**

 一般使用Mutistep策略，搁几个epoch调整一次学习率的数量级。

#### cyclical learning rate

**实验证明通过设置上下界，让学习率在其中进行变化，可以在模型迭代的后期更有利于克服因为学习率不够而<big>无法跳出鞍点</big>的情况。**

确定学习率上下界的方法则可以使用LR range test方法，即使用不同的学习率得到精度曲线，然后获得精度升高和下降的两个拐点，或者将精度最高点设置为上界，下界设置为它的1/3大小。

<img src="https://pic2.zhimg.com/80/v2-d93b708210d81d896e3d0b8349b3d66d_720w.jpg" alt="img" style="zoom: 50%;" />

SGDR方法则是比cyclical learning rate变换更加平缓的周期性变化方法，如下图，效果与cyclical learning rate类似。

<img src="https://pic2.zhimg.com/80/v2-317cd6d9bbaa0c6a7b74016a03f68d1d_720w.jpg" alt="img" style="zoom:50%;" />

### 自适应学习率变换方法

也就是Adam，详情看《<u>10_理清深度学习优化函数发展脉络.md</u>》

## Batchsize如何影响模型性能？

模型性能对batchsize虽然没有学习率那么敏感，但是在进一步提升模型性能时，batchsize就会成为一个非常关键的参数。

**3.1 大的batchsize减少训练时间，提高稳定性**

这是肯定的，同样的epoch数目，<u>大的batchsize需要的batch数目减少了，所以可以减少训练时间（前提是batchsize大，lr也要大，确保一个epoch参数更新的总长度不变）</u>，目前已经有多篇公开论文在1小时内训练完ImageNet数据集。另一方面，**大的batch size梯度的计算更加稳定，因为模型训练曲线会更加平滑**。在微调的时候，大的batch size可能会取得更好的结果。

**3.2 大的batchsize导致模型泛化能力下降**

在一定范围内，增加batchsize有助于收敛的稳定性，但是随着batchsize的增加，模型的性能会下降，如下图，来自于文[5]。

<img src="https://pic1.zhimg.com/80/v2-e9ebb41acf19d7502646e99b909a0bd4_720w.jpg" alt="img" style="zoom:50%;" />

主要原因是小的batchsize带来的噪声有助于逃离局部最优。

**3.3 小结**

**batchsize在变得很大(超过一个临界点)时，会降低模型的泛化能力**。在此临界点之下，模型的性能变换随batch size通常没有学习率敏感。

## 学习率和BatchSize的关系

**通常当我们增加batchsize为原来的N倍时，学习率应该增加为原来的N倍，因为要保证经过同样的样本后更新的权重相等，**

- **如果增加了学习率，那么batch size最好也跟着增加，这样收敛更稳定。**
- **尽量使用大的学习率，因为很多研究都表明更大的学习率有利于提高泛化能力。**如果真的要衰减，可以尝试其他办法，比如增加batch size，学习率对模型的收敛影响真的很大，慎重调整。

## 问题

### 10k样本用SGD，batchsize取1，100，10000哪个收敛最快？

100。

1. 大的bs，一个epoch需要的iter次数就会变少，又一般有5次bs=1比1次bs=5的时间长，所以bs越大训练速度越快。
   而且，bs越大，梯度更加稳定，loss曲线更加平滑。
2. bs过大，导致模型泛化性下降，因为小bs更容易跳出局部最优点（sharp minimum）。一般bs不能超过8000。

因此，太大，太小都不好。

![img](https://pic3.zhimg.com/80/v2-0c8205e13546eb21f0739f156fc083f6_720w.jpg)

## 参考链接

[【AI不惑境】学习率和batchsize如何影响模型的性能？](https://zhuanlan.zhihu.com/p/64864995#:~:text=batchsize%E5%9C%A8%E5%8F%98%E5%BE%97%E5%BE%88%E5%A4%A7%20%28%E8%B6%85%E8%BF%87%E4%B8%80%E4%B8%AA%E4%B8%B4%E7%95%8C%E7%82%B9%29%E6%97%B6%EF%BC%8C%E4%BC%9A%E9%99%8D%E4%BD%8E%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B%E3%80%82%20%E5%9C%A8%E6%AD%A4%E4%B8%B4%E7%95%8C%E7%82%B9%E4%B9%8B%E4%B8%8B%EF%BC%8C%E6%A8%A1%E5%9E%8B%E7%9A%84%E6%80%A7%E8%83%BD%E5%8F%98%E6%8D%A2%E9%9A%8Fbatch,size%E9%80%9A%E5%B8%B8%E6%B2%A1%E6%9C%89%E5%AD%A6%E4%B9%A0%E7%8E%87%E6%95%8F%E6%84%9F%E3%80%82%20%E9%80%9A%E5%B8%B8%E5%BD%93%E6%88%91%E4%BB%AC%E5%A2%9E%E5%8A%A0batchsize%E4%B8%BA%E5%8E%9F%E6%9D%A5%E7%9A%84N%E5%80%8D%E6%97%B6%EF%BC%8C%E8%A6%81%E4%BF%9D%E8%AF%81%E7%BB%8F%E8%BF%87%E5%90%8C%E6%A0%B7%E7%9A%84%E6%A0%B7%E6%9C%AC%E5%90%8E%E6%9B%B4%E6%96%B0%E7%9A%84%E6%9D%83%E9%87%8D%E7%9B%B8%E7%AD%89%EF%BC%8C%E6%8C%89%E7%85%A7%E7%BA%BF%E6%80%A7%E7%BC%A9%E6%94%BE%E8%A7%84%E5%88%99%EF%BC%8C%E5%AD%A6%E4%B9%A0%E7%8E%87%E5%BA%94%E8%AF%A5%E5%A2%9E%E5%8A%A0%E4%B8%BA%E5%8E%9F%E6%9D%A5%E7%9A%84N%E5%80%8D%20%5B5%5D%E3%80%82)