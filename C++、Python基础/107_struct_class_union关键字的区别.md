## struct、class、union的区别

1. 访问权限

   1. <u>struct、union的默认访问权限都是public</u>
   2. class的默认访问权限是private。但二值都可以通过显示的指定访问权限。

2. 内存大小：

   1. struct、class的内存大小为所有成员的内存之和
   2. 而union的内存大小为最大的成员的内存大小，成员变量之间共享内存。

3. 继承：

   1. struct、class都可以进行继承与被继承，不过struct只能添加带参数的构造函数。
   2. union不可以作为基类，也不能从别的基类派生。

4. 成员：

   1. union不能包含虚函数，静态数据变量，**也不能存放带有构造、析构、拷贝构造等函数的类**。(比如就不能添加vector变量)

5. template模板

   1. <u>class可以使用模板，而struct不可以</u>

      ```cpp
      template <typename T> 或者 template <class T>
      ```

      

## union

### 概念

C++ union结构式一种节省空间的特殊的类，一个union可以有多个数据成员，但是==在任意时刻只有一个数据成员可以有值，当某个成员被赋值后，其他成员变为未定义状态==。
特点

1. Union中得**默认访问权限是public**，但可以为其成员设置public、protected、private权限。
2. 能够包含访问权限、成员变量、成员函数（可以包含构造函数和析构函数）。
3. 不能包含虚函数、静态数据变量和引用类型的变量。
4. 它也**不能被用作其他类的基类**，它本身也不能有从某个基类派生而来。
5. 联合里**不允许存放带有构造函数、析够函数、复制拷贝操作符等的类**，因为他们共享内存，编译器无法保证这些对象不被破坏，也无法保证离开时调用析够函数。

**（尽量不要让union带有对象）**

### 例子

在C/C++程序的编写中，==当多个基本数据类型或复合数据结构要占用同一片内存时==，我们要使用联合体；当多种类型，多个对象，多个事物只取其一时（我们姑且通俗地称其为**“n 选1**”），我们也可以使用联合体来发挥其长处。

在任意时刻，联合中只能有一个数据成员可以有值。当给联合中某个成员赋值之后，该联合中的其它成员就变成未定义状态了。

```C++
#include <iostream>
using namespace std;

union Test {
	struct {
		int x;
		int y;
		int z;
	}s;
	int k;
}myUnion;

int main()
{
	myUnion.s.x = 4;
	myUnion.s.y = 5;
	myUnion.s.z = 6;
	myUnion.k = 0;
	cout << myUnion.s.x << endl;
	cout << myUnion.s.y << endl;
	cout << myUnion.s.z << endl;
	cout << myUnion.k << endl;
	cout << &myUnion.s.x << endl;
	cout << &myUnion.s.y << endl;
	cout << &myUnion.s.z << endl;
	cout << &myUnion.k << endl;
	return 0;
}
```

输出结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200708105947537.png)

**<big>解释</big>**
　　==union类型是共享内存的，以最大的成员的内存大小作为自己的大小==。每个数据成员在内存中得其实地址是相同的。myUnion这个结构就包含s这个结构体，而大小也等于s这个结构体的大小，**在内存中的排列为声明的顺序x,y,z从低到高**，然后赋值的时候，在内存中，就是x的位置放置4，y的位置放置5，z的位置放置6，现在对k赋值，**对k的赋值因为是union，要共享内存，所以从union的首地址开始放置**，首地址开始的位置其实是x的位置，这样原来内存中x的位置就被k所赋的值代替了，就变为0了，这个时候要进行打印，就直接看内存里就行了，**x的位置也就是k的位置是0，而y，z的位置的值没有改变**。

## 参考链接

[C++中Union学习笔记](https://blog.csdn.net/sinat_22078359/article/details/107201249)（总结较多，适合面试回答看）

[C++中union的使用方法](https://blog.csdn.net/hou09tian/article/details/80816445)（非常详细，例子清楚，可以仔细看）