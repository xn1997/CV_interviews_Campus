## RANSAC（随机一致性采样）

**优点：对噪声点不敏感，**因为他不对噪声点进行建模。
而最小二乘法对所有点进行建模，所以他对噪声很敏感。

### 过程

总的来说：类似GMM、KNN这种算法，都是先随机挑一部分数据来拟合模型，然后计算其他数据与该模型的距离从而引入一些新的类内数据，再用新的类内数据来拟合出新的模型，迭代这个过程，直到有足够多的点被归为类内点。
本质上都是迭代算法，使用EM算法进行学习，E步固定模型分类数据，M步固定数据更新模型。

流程概括：

<img src="https://pic1.zhimg.com/80/v2-92f0ad1a9054d4bd19a759b7e3167bcc_720w.jpg" alt="img" style="zoom: 67%;" />

举个例子：使用RANSAC——拟合直线

<img src="https://pic2.zhimg.com/80/v2-67e966c92f04f232010255dc5cd1b92d_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic1.zhimg.com/80/v2-3693478f142577031cfc29b9d61e58c8_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic3.zhimg.com/80/v2-bd7445a60766817022f8506274f2eeba_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic2.zhimg.com/80/v2-fcd467425195baccd67f7d8ec6101c2d_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic1.zhimg.com/80/v2-7225d7e8e5dd5d6ea19aa560c866dd9c_720w.jpg" alt="img" style="zoom:50%;" />

<img src="https://pic2.zhimg.com/80/v2-959cf86f0907368c4acc60c6d43d22d5_720w.jpg" alt="img" style="zoom:50%;" />

### 参考链接

[计算机视觉基本原理——RANSAC](https://zhuanlan.zhihu.com/p/45532306)

