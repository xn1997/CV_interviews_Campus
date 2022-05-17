普通的线性回归的损失函数就是MSE，即

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta%5E%2A+%3D+argmin_%7B%5Cbeta%7D+%5Cfrac%7B1%7D%7Bn%7D%5Csum_%7Bi+%3D+0%7D%5En%5Cleft%28+y_i+-+%5Cbeta%5ETx_i+%5Cright%29%5E2+%5C%5C)

Lasso是添加了L1正则化

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta%5E%2A+%3D+argmin_%7B%5Cbeta%7D+%5Cfrac%7B1%7D%7Bn%7D%5CVert+y+-+X%5Cbeta+%5CVert_2%5E2+%2B+%5Clambda+%5CVert+%5Cbeta+%5CVert_1+%5C%5C)

Ridge Regression是添加了L2正则化

![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta%5E%2A+%3D+argmin_%7B%5Cbeta%7D+%5Cfrac%7B1%7D%7Bn%7D%5CVert+y+-+X%5Cbeta+%5CVert_2%5E2+%2B+%5Clambda+%5CVert+%5Cbeta+%5CVert_2%5E2+%5C%5C)

**L1正则化会使权重变的稀疏，而L2正则化是权值衰减**，使权值趋向于0，即Weight Decay。

**L1正则化技术又称为 Lasso Regularization。**



L1和L2与贝叶斯先验的关系：

L1表示权值服从均值为0的**拉普拉斯分布**。
L2表示权值服从均值为0的**高斯分布**。

![[公式]](https://www.zhihu.com/equation?tex=p_%7Bl2%7D%28%5Cbeta_j%29+%3D+%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi+%5Calpha%7D%7D%5Cexp+%5Cleft%28+-%5Cfrac%7B%5Cbeta_j%5ET%5Cbeta_j%7D%7B2%5Calpha%7D%5Cright%29+%5C%5C+p_%7Bl1%7D%28%5Cbeta_j%29+%3D+%5Cfrac%7B1%7D%7B2%5Calpha%7D%5Cexp+%5Cleft%28+-%5Cfrac%7B%7C%5Cbeta_j%7C%7D%7B%5Calpha%7D%5Cright%29+%5C%5C+)

### 参考链接

[从Lasso开始说起](https://zhuanlan.zhihu.com/p/46999826)