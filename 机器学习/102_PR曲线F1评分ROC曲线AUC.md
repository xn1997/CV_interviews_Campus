参考：《百面机器学习》

# PR曲线

> TP（ True Positive）：真正例
>
> FP（ False Positive）：假正例
>
> FN（False Negative）：假反例
>
> TN（True Negative）：真反例

**精确率（Precision）:**	$Precision=\frac{TP}{TP+FP}$ 

**召回率（Recall）:**	$Recall=\frac{TP}{TP+FN}$ 

> 精确率（Y），也称为查准率，衡量的是**分类正确的正样本个数占分类器判定为正样本的样本个数的比例**。
>
> 召回率（X），也称为查全率，衡量的是**分类正确的正样本个数占真正的正样本个数的比例。**

## F1-score

F1 score是分类问题中常用的评价指标，定义为精确率（Precision）和召回率（Recall）的调和平均数。
$$
F1=\frac{2}{\frac{1}{Precision}+\frac{1}{Recall}}=\frac{2×Precision×Recall}{Precision+Recall}
$$

F1 score 综合考虑了精确率和召回率，其结果更偏向于 Precision 和 Recall 中较小的那个。可以更好地反映模型的性能好坏，而不是像算术平均值直接平均模糊化了 Precision 和 Recall 各自对模型的影响。

> 补充另外两种评价方法：

**加权调和平均：**

上面的 F1 score 中， Precision 和 Recall 是同等重要的，而有的时候可能希望我们的模型更关注其中的某一个指标，这时可以使用加权调和平均：
$$
F_{\beta}=(1+\beta^{2})\frac{1}{\frac{1}{Precision}+\beta^{2}×\frac{1}{Recall}}=(1+\beta^{2})\frac{Precision×Recall}{\beta^{2}×Precision+Recall}
$$
当 $\beta > 1$ 时召回率有更大影响， $\beta < 1$ 时精确率有更大影响， $\beta = 1$ 时退化为 F1 score。

即**F2，召回率权重更高，F0.5，精确率权重更高。**
（根据式子可以看出$\beta$越大，P就可以小一些，也就是不重要）



**几何平均数：**
$$
G=\sqrt{Precision×Recall}
$$

# ROC曲线

**真阳性率（True Positive Rate）：**$TPR =\frac{TP}{P}=\frac{TP}{TP+FN}$

**假阳性率（False Positive Rate，FPR）：**$FPR =\frac{FP}{N}=\frac{FP}{FP+TN}$

> 真阳性率（Y），也称为灵敏度，衡量的是**分类正确的正样本个数占真正的正样本个数的比例**。
>
> 假阳性率（X），也称为特异度，衡量的是**错误识别为正例的负样本个数占真正的负样本个数的比例。**

![img](https://pic1.zhimg.com/80/v2-383b1279e560ca96c85204ccaf564037_720w.jpg?source=1940ef5c)

## AUC

参考链接：[AUC的计算方法及相关总结](https://blog.csdn.net/renzhentinghai/article/details/81095857)
$$
A U C=\frac{\sum_{i \in \text { positiveclass }} \operatorname{rank}_{i}-\frac{M(1+M)}{2}}{M \times N}
$$
AUC指的是ROC曲线下的面积大小，该值能够反映基于ROC曲线衡量出的模型性能。**AUC越大，说明分类器越可能把真正的正样本排在前面，分类性能越好。**

### 参考链接

[如何理解机器学习和统计中的AUC？](https://www.zhihu.com/question/39840928)

# PR和ROC对比

- 当正负样本的分布发生变化时，ROC曲线的形状能够基本保持不变，而P-R曲线的形状一般会发生较剧烈的变化。

  **ROC曲线能够尽量降低不同测试集带来的干扰**，在正负样本不平衡时，更能反映**模型本身的特性**。

- 如果研究者希望更多地看到**模型在特定数据集**上的表现，P-R曲线则能够更直观地反映其性能。