### label smooth

**目的：**

通常情况下的标签都是`[0,0,1,0,0,0]`这种非常极端的约束，为了缓和label对网络的约束，提高模型的**泛化能力，减少过拟合**的风险。

如果使用硬标签，那么当有标注错误的样本时，会产生较大的损失，而软标签则可以对于这些噪声样本产生抑制，防止其产生较大的损失。（**不希望模型对其预测结果过度自信**）

**方法：**

对标签做一个平滑处理，公式如下：
$$
q_i=
\begin{cases}
1-\frac{N-1}{N}\epsilon=1-\epsilon+\frac{\epsilon}{N}&if\ i=y\\
\frac{\epsilon}N&otherwise
\end{cases}
$$
也就是说，若`label=[0,0,1,0,0,0]`，那么N=5（这里label包含了背景类，所以有6维）。若定义$\epsilon=0.1$，那么带入上面公式就可以得到，平滑后的label为`[0.02，0.02，0.9，0.02，0.02，0.02]`。这些是软标签，而不是硬标签，即0和1。**其实更新后的分布就相当于往真实分布中加入了噪声**。

```python
#---------------------------------------------------#
#   平滑标签
#---------------------------------------------------#
def smooth_labels(y_true, label_smoothing,num_classes):
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes
```



**交叉熵损失发生变化**

label smoothing将交叉熵损失函数作如下改变：

  <img src="https://img-blog.csdnimg.cn/20190128232406554.png" alt="img" style="zoom:60%;" /> 

与之对应，label smoothing将最优的预测概率分布作如下改变：

  <img src="https://img-blog.csdnimg.cn/20190128232827323.png" alt="img" style="zoom:60%;" /> 

阿尔法可以是任意实数，最终**通过抑制正负样本输出差值，使得网络能有更好的泛化能力**。

# 参考

[深度学习 | 训练网络trick——label smoothing(附代码)](https://blog.csdn.net/qiu931110/article/details/86684241)