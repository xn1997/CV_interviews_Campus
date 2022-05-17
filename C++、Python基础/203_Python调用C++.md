使用**ctypes**模块直接加载so

函数为：`ctypes.cdll.LoadLibrary("so文件路径")`

### 生成so库

将C++编译成so即可，使用cmake就行，不过记得一定要使用`extern "c" {函数}`将程序以c的形式编译，不然Python无法调用。

**准备C语言程序，保存为add.c**

```c
#include <stdio.h>
 
int add_int(int, int);
float add_float(float, float);
 
int add_int(int num1, int num2)
{
    return num1 + num2;
}
 
float add_float(float num1, float num2)
{
    return num1 + num2;
}
```

### python调用so库

```python
import ctypes
 
#load the shared object file
adder = ctypes.cdll.LoadLibrary('./adder.so')
 
#Find sum of integers
res_int = adder.add_int(4,5)
print("4 + 5 = " + str(res_int))
 
#Find sum of floats
a = ctypes.c_float(5.5)
b = ctypes.c_float(4.1)
 
add_float = adder.add_float
add_float.restype = ctypes.c_float
 
print("5.5 + 4.1 = " + str(add_float(a, b)))
```

### 参考链接

[Python调用C/C++库](https://blog.csdn.net/u013171226/article/details/109806072)

