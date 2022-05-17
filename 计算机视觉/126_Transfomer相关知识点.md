## Self-Attention

### 3.2 Q, K, V 的计算

Self-Attention 的输入用矩阵X进行表示，则可以使用线性变阵矩阵**WQ,WK,WV**计算得到**Q,K,V**。计算如下图所示，**注意 X, Q, K, V 的每一行都表示一个单词。**

<img src="https://pic3.zhimg.com/80/v2-4f4958704952dcf2c4b652a1cd38f32e_720w.jpg" alt="img" style="zoom:50%;" />

### 3.3 Self-Attention 的输出

得到矩阵 Q, K, V之后就可以计算出 Self-Attention 的输出了，计算的公式如下：

![img](https://pic2.zhimg.com/80/v2-9699a37b96c2b62d22b312b5e1863acd_720w.jpg)

公式中计算矩阵**Q**和**K**每一行向量的内积，为了防止内积过大，因此除以 ![[公式]](https://www.zhihu.com/equation?tex=d_%7Bk%7D) 的平方根。**Q**乘以**K**的转置后，得到的矩阵行列数都为 n，n 为句子单词数，这个矩阵可以表示单词之间的 attention 强度。下图为**Q**乘以 ![[公式]](https://www.zhihu.com/equation?tex=K%5E%7BT%7D) ，1234 表示的是句子中的单词。

<img src="https://pic2.zhimg.com/80/v2-9caab2c9a00f6872854fb89278f13ee1_720w.jpg" alt="img" style="zoom:50%;" />

得到![[公式]](https://www.zhihu.com/equation?tex=QK%5E%7BT%7D) 之后，使用 Softmax 计算每一个单词对于其他单词的 attention 系数，公式中的 Softmax 是对矩阵的每一行进行 Softmax，即每一行的和都变为 1.

<img src="https://pic1.zhimg.com/80/v2-96a3716cf7f112f7beabafb59e84f418_720w.jpg" alt="img" style="zoom:50%;" />

得到 Softmax 矩阵之后可以和**V**相乘，得到最终的输出**Z**。

<img src="https://pic4.zhimg.com/80/v2-7ac99bce83713d568d04e6ecfb31463b_720w.jpg" alt="img" style="zoom:50%;" />

上图中 Softmax 矩阵的第 1 行表示单词 1 与其他所有单词的 attention 系数，最终单词 1 的输出 ![[公式]](https://www.zhihu.com/equation?tex=Z_%7B1%7D) 等于所有单词 i 的值 ![[公式]](https://www.zhihu.com/equation?tex=V_%7Bi%7D) 根据 attention 系数的比例加在一起得到，如下图所示：

<img src="https://pic3.zhimg.com/80/v2-27822b2292cd6c38357803093bea5d0e_720w.jpg" alt="img"  />

# Swin-Transformer代码解析

<img src="https://pic4.zhimg.com/80/v2-dcaeae4ffe43119517fc4ac30e698c53_720w.jpg" alt="img" style="zoom:50%;" />

## 预处理PatchEmbed

```mermaid
graph TD
1(N,3,224,224)--conv-3,96,4,4-->N,96,56,56--flaten+transpose-->N,56*56,96-->dropout
2(N,C,H,W)--conv-C,C2,4,4-->N,C2,H/4,W/4--flaten+transpose-->N,H/4*W/4,C2-->dropout
```

## stage

<u>**每个stage的输入和输出都是`N,HW,C`的形式，因此，最后的输出也相当于是一个特征图，与其他的backbone没有任何区别。**</u>

### PatchMerging

就是focus结构，替代池化使用的

<img src="https://pic4.zhimg.com/80/v2-f9c4e3d69da7508562358f9c3f683c63_720w.png" alt="img" style="zoom:100%;" />

以下的`N,H*W,C`等价于上一节的`N,H/4*W/4,C2`

```mermaid
graph TD
N,H*W,C--focus池化-->N,H/2*W/2,4*C--norm+liner-4C,2C-->N,H/2*W/2,2C
```

### block



```mermaid
graph TD
N,H*W,C--LN+reshape-->N,H,W,C--W-MSA/SW-MSA-->N,H/2*W/2,C
```

### W-MSA

**主要核心还是self-attention，只不过是局部使用而已，减少计算量。其实就是nonlocal**

### SW-MSA

## 参考链接

[Swin-Transformer结合代码深度解析](https://zhuanlan.zhihu.com/p/384514268)

[搞懂 Vision Transformer 原理和代码，看这篇技术综述就够了（十六）](https://mp.weixin.qq.com/s/EmplGLcnvjE6SN5WY5Cg2w)——介绍的非常详细（主要看这个，也有VIT的讲解）