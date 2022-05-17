## Transformer

Transformer的Encoder-Decoder结构

<img src="https://pic1.zhimg.com/80/v2-5a252caa82f87920eadea2a2e93dc528_720w.jpg" alt="img" style="zoom:50%;" />

Encoder的结构图

<img src="https://pic3.zhimg.com/80/v2-89e5443635d7e9a74ff0b4b0a6f31802_720w.jpg" alt="img" style="zoom: 80%;" />

FFN就是两层的全连接层，只不过第一层的激活函数是ReLU，第二层是线性激活函数（也就是没有激活函数）

#### 参考链接

[详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)

## DERT

<img src="https://pic2.zhimg.com/80/v2-fde02e0549a2f911075e2ffef5892b4d_720w.jpg" alt="img" style="zoom:200%;" />

1. 首先，输入图片( $3\times{H_o}\times{W_o}$ )经过 CNN backbone 得到分辨率较低的 feature maps ( ![[公式]](https://www.zhihu.com/equation?tex=C%5Ctimes%7BH%7D%5Ctimes%7BW%7D) ), 
2. 然后进入到 Transformer 的 Encoder 部分，首先用 1x1 的 conv 把输入的 C 给降维到较小的 d ( ![[公式]](https://www.zhihu.com/equation?tex=d%5Ctimes%7BH%7D%5Ctimes%7BW%7D) ) 并 reshape 到 ![[公式]](https://www.zhihu.com/equation?tex=d%5Ctimes%7BHW%7D) （这里就和NLP中的输入一样的，相当于HW个词，每个词的特征是d维度)
3. 将$d×HW$添加位置编码，送入标准的Transformer结构即可。最后 decoder 的输出给到 N 个共享权重的 FFN 得到预测目标的类别和坐标信息。

唯一的区别就是，head换成了transformer，即利用全局信息来回归出每一个目标。（这里其实是假定好最多有N个目标的）