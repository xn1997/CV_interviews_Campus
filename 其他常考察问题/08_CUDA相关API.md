## cublas实现Einsum

### cublasSgemm使用说明

C++下的矩阵运算，应该使用cublas库，其使用了GPU进行加速。
普通的矩阵读取都是按行读按行存，而**cublas的矩阵是按行读按列存**，那么就导致数据的维度会出现问题，因此需要使用相应的操作以使其可以正常计算。

**API参数**

该函数实现的计算为$C = alpha*A*B + beta*C(看起来没啥问题，使用起来呵呵哒)$
但重要的是其中$A*B$的计算方法是很特别的，需要注意，以下进行解释。

```cpp
cublasSgemm{
    cublasHandle_t handle, // 调用 cuBLAS 库时的句柄
    cublasOperation_t transa, // 是否对A转置
    cublasOperation_t transb, // 是否对B转置
    int m, int n, int k, // mnk表示矩阵计算时候的维度
    const float *alpha,
    const float *A, //矩阵A
    int lda,  //按行读取的长度
    const float *B, 
    int ldb, 
    const float *beta,
    float *C, 
    int ldc // 按列取的个数
    }
```

**API使用示例**

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2 ,4, 3, &alpha, *A, 2, *B, 3, &beta, *C, 4);
```



![img](https://img-blog.csdnimg.cn/20200403200903415.png)

- `int m,int n, int k`这三个参数，这三个参数**依次**代表的是B1的行，B1的列，A1的列（B1的行）。也就是最终参与运算的数据大小。
- `lda`：将A**按行读**，按**每`lda`个数一列**排好，得到A1
- `lda`：同`lda`
- `ldc`：将C1**按列读**，按**每`lda`个数一行**排好，得到C
- `transa/transb`：是否对输入数据转置，转置的位置是在上图的A1和B1之后。

可以理解为：C++/C是以行优先进行读取和存储，而这里的cublas是以列优先进行读取和存储。所以读入数据的时候是按行读（C++），按列存（GPU）；输出数据时是按列读（GPU），按行存（C++）

### 性质（如何使用）

由于cuBLAS特有的列优先机制，因此通过合理的设置`lda/ldb/ldc`可以**达到转置效果**：
设置`lda`为A的列数，则$A_1=A^T$
设置`ldb`为B的列数，则$B_1=B^T$
设置`ldc`为C的行数，则$C=C_1^T$

```cpp
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 3, &alpha, *B, 4, *A, 3 , &beta, *C, 4);
```

![img](https://img-blog.csdnimg.cn/20200403210129584.png)

因此，根据该性质就可以推导出A×B的计算方法为
$$
C=AB=(B^TA^T)^T
$$
就是**想办法凑出输入的逆、输出的逆**，这样就可以使用该函数进行矩阵计算了。

### 参考链接

[cuBLAS矩阵乘法](https://blog.csdn.net/feng__shuai/article/details/105299959)

## cudaMemcpy与cudaMemcpyAsync的区别(FIXME+)

参考链接：[cudaMemcpy与cudaMemcpyAsync的区别](https://www.cnblogs.com/shrimp-can/p/5231857.html)

1. 前者同步，后者异步

## 参考链接

[CUDA编程之快速入门 - Madcola - 博客园](https://www.cnblogs.com/skyfsm/p/9673960.html)