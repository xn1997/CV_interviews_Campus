## 迭代器

### `__getitem__`

凡是在类中定义了这个`__getitem__ `方法，那么它的实例对象（假定为p），可以像这样p[key] 取值，当实例对象做p[key] 运算时，会调用类中的方法`__getitem__`。

- **一般如果想使用索引访问元素时，就可以在类中定义这个方法`(__getitem__(self, key))`。**

- 还可以用在对象的迭代上

  ```python
  class STgetitem:
  
      def __init__(self, text):
          self.text = text
  
      def __getitem__(self, index):
          result = self.text[index].upper()
          return result
  
  
  p = STgetitem("Python")
  print(p[0])
  print("------------------------")
  for char in p:
      print(char)
  
  # output
  P
  ------------------------
  P
  Y
  T
  H
  O
  N
  ```

### `__len__`

如果一个类表现得像一个list，要获取有多少个元素，就得用` len() `函数。

要让 `len()` 函数工作正常，类必须提供一个特殊方法`__len__()`，它返回元素的个数。

例如，我们写一个 Students 类，把名字传进去：

```python
class Students(object):
    def __init__(self, *args):
        self.names = args
    def __len__(self):
        return len(self.names)
```

只要正确实现了`__len__()`方法，就可以用`len()`函数返回Students实例的“长度”：

```python
>>> ss = Students('Bob', 'Alice', 'Tim')
>>> print len(ss)
3
```

### 迭代器`__iter__()`和`__next()__`

`__iter__()`:迭代器，生成迭代对象时调用**（只在生成对象时调用一次）**，返回值必须是对象自己,然后for可以循环调用next方法；**如果对同一个对象生成两个迭代器对象，那么这两个迭代器对象其实是一个，对一个修改，另一个也会自动执行相应的修改**

`__next()__`:每一次for循环都调用该方法（优先级高于索引`__getitem__`）

```python
class A(object):
    def __init__(self,num):
        self.num = num
        self.start_num = -1
    
    def __iter__(self):
        '''
        @summary: 迭代器，生成迭代对象时调用，返回值必须是对象自己,然后for可以循环调用next方法
        '''
        print "__iter__"
        return self
    
    def __next__(self):
        '''
        @summary: 每一次for循环都调用该方法（必须存在）
        '''
        self.start_num += 1
        if self.start_num >= self.num:
            raise StopIteration()
        return self.start_num
    
if __name__ == "__main__":
    for i in A(10):
        print i,
 # >>>>>>>>>>>>输出>>>>>>>>>>>
 __iter__
 0 1 2 3 4 5 6 7 8 9
```

<font color=red><big>**`for i in A:`的本质**</big></font>

```python
for i in A:
	...
"""
本质是
1. 自动使用iter(A),执行A类的__iter__()方法为可迭代对象A创建一个迭代器tmp_iter
2. 每次循环（迭代）都调用一次next(tmp_iter)执行A类对应的__next__()方法将返回值赋予i
"""
```



### 在深度学习中的运用

`__getitem__`：常用于数据加载类（Dataset）中，用于对象的迭代，每次迭代都执行一次`__getitem__`中的数据加载函数，以获取相应的数据

`__len__`：常用于数据加载类（Dataset）中，以便可以使用len()函数获取数据总数，方便计算训练一个epoch需要多个batch。（更多的是显示作用）

`__iter__()`：常用于数据加载类（Dataset）中，固定写法返回self

`__next()__`和`__getitem__`很像，深度学习中一般只使用`__getitem__`不断读取数据，`next()`使用较少

### 三者的区别**

1. 都可以在for中使用，进行迭代操作
2. `__getitem__`用于可迭代对象的索引（如p[key]），也可用于for的迭代（但优先级低于`__next__`）
3. `__iter__和__next__`只用于对象的迭代（如for）（通过`iter()/next()`调用对应的创建迭代器、迭代操作函数）
4. 可迭代对象必须包含`__getitem__`或者同时包含`__iter__和__next__`，或者同时包含。（同时包含时，在循环中`__next__`的优先级高于`__getitem__`，只会执行`__next__`）

### 参考资料

[__getitem__、__iter__、 __next__方法的使用](https://blog.csdn.net/m0_46653437/article/details/108743767)（都是例子，重点参考`__getitem__`）

[迭代、可迭代对象，迭代器、生成器、装饰器及yield与return的区别](https://blog.csdn.net/hhhhhh5863/article/details/88746431)

## 生成器

使用了 **yield** 的函数被称为生成器（generator）。

生成器是一个返回迭代器的函数，**只能用于迭代操作**，更简单点理解生成器就是一个迭代器。

调用生成器运行的过程中，每次**遇到 yield 时函数会暂停并保存当前所有的运行信息，返回 yield 的值**, 并在下一次**执行 next() 方法时从当前位置继续运行**

调用一个生成器函数，**返回的是一个迭代器对象**。

### 参考链接

[Python3 迭代器与生成器](https://www.runoob.com/python3/python3-iterator-generator.html)（生成器介绍）

## 装饰器

**实现的技术前提**

1. **高阶函数**：函数参数是一个函数名或者返回的是函数名
2. **函数嵌套**：在一个函数中又定义了一个新的函数
3. **闭包**：在函数嵌套中，内部函数对外部函数作用域变量的引用(除去全局变量),则称内部函数为闭包。

**装饰器，顾名思义，就是增强函数或类的功能的一个函数。**（对函数进行功能更新，主要是前后处理，无法对函数内部进行修改）

### **使用方法**

假设decorator是定义好的装饰器。

方法一：不用@符号

```python
# 装饰器不传入参数时 
f = decorator(函数名) 
# 装饰器传入参数时
f = (decorator(参数))(函数名)
```

方法二：采用@符号

````python
# 已定义的装饰器
@decorator 
def f():  
    pass

# 执行被装饰过的函数
f()
````

**装饰器可以传参，也可以不用传参。**

自身不传入参数的装饰器（采用两层函数定义装饰器）

```python
def login(func) :
    def wrapper(*args ,**kargs):
    	print( '函数名:%s '% func. __name__)
        return func(*args,**kargs)
    return wrapper

@login
def f():
	print( " inside decorator ! ')
          
f()
# 输出:
# >>函数名:f
# >>函数本身:inside decorator !
```

自身传入参数的装饰器（采用三层函数定义装饰器）

```python
def login(text):
    def decorator(func) :
        def wrapper(*args,**kargs ):
        print( "%s----%s "%(text， func.__name__))
        return func(*args ,**kargs)
        return wrapper
    return decorator
# 等价于=->( login(text))(f)==>返回wrapper
@login( 'this is a parameter of decorator' )
def f():
	print( ' 2019-06-13')
#等价于-=>(login(text))(f)()==>调用wrapper()并返回f()
f()

# 输出:
# => this is a parameter of decorator-- --f
# =>2619-e6-13

```

==本质上@就是用装饰器来装饰@下方的那个函数==

### 内置装饰器:

##### `@property`

**把类内方法当成属性来使用**，必须要有返回值，相当于getter；

假如没有定义 `@func.setter `修饰方法的话，就是只读属性，不可修改。

```python
"以下就是将类内方法，装饰成属性值（属性值就是类内方法的返回值）"
class Car:
    def __init__(self, name, price):
    self._name = name
    self._price = price
    
    @property
    def car_name(self):
        return self._name	
    
    # car_name可以读写的属性，以下是固定写法，如果不加，car_name就是一个只读属性
    @car_name.setter
    def car.name(self, value):
    	self._name = value
        
    # car_price是只读属性
    @property
    def car_price(self):
    	return str(self._price) + '万'
    
benz = Car('benz', 30)
print(benz.car_name ) # benz
benz.car_name = "baojun'
print(benz.car_name ) # baojun
print(benz.car_price) # 30万

```

##### **`@classmethod`**

**类方法**，不需要self参数，但第一个参数需要是**表示自身类的cls参数**。

##### **`@staticmethod`**

**静态方法**，不需要表示自身对象的self和自身类的cls参数，就跟使用函数一样。

总结：二者其实就相当于将类方法变成可以直接使用类调用的函数，**等同于C++定义静态成员函数**。

不同点：
**`@staticmethod`**不需要传入如何参数，但内部使用类方法时，需要把类名写上。
**`@classmethod`**需要传入一个代替类名的cls参数，内部使用类方法时，直接使用cls代替类名即可。
**`@classmethod`相比`@staticmethod`更加方便**，因为如果类名很长，前者依然只需要写cls即可，而后者就得把很长的类名写上了。

```python
class Demo(object):
    text = "三种方法的比较"
    
    def instance_method(self):
    	print("调用实例方法")
        
    @classmethod
    def class_method(cls):
        print("调用类方法")
        print("在类方法中 访问类属性 text: {}".format(cls.text))
        print("在类方法中 调用实例方法 instance_method: {}".format(cls().instance_method()
                                                        
    @staticmethod
    def static_method():
        print("调用静态方法")
        print("在静态方法中 访问类属性 text: {}".format(Demo.text))
        print("在静态方法中 调用实例方法 instance_method: {}".format(Demo().instance_method
if __name__ == "__main__":
    # 实例化对象
    d = Demo()
                                                         
    # 对象可以访问 实例方法、类方法、静态方法
    # 通过对象访问text属性
    print(d.text)
                                                         
    # 通过对象调用实例方法
    d.instance_method()
                                                         
    # 通过对象调用类方法
    d.class_method()
                                                         
    # 通过对象调用静态方法
    d.static_method()
                                                         
    # 类可以访问类方法、静态方法
    # 通过类访问text属性
    print(Demo.text)
                                                         
    # 通过类调用类方法
    Demo.class_method()
                                                         
    # 通过类调用静态方法
    Demo.static_method()
```

##### classmethod与staticmethod

这两个方法的用法是类似的，在上面的例子中大多数情况下，classmethod也可以通过staticmethod代替，在通过类调用时，这两者对于调用者来说是不可区分的。

这两者的区别在于，classmethod增加了一个对实际调用类的引用，这带来了很多方便的地方：

1. 方法可以判断出自己是通过基类被调用，还是通过某个子类被调用
2. 通过子类调用时，方法可以返回子类的实例而非基类的实例
3. 通过子类调用时，方法可以调用子类的其他classmethod

一般来说classmethod可以完全替代staticmethod。staticmethod唯一的好处是调用时它返回的是一个真正的函数，而且每次调用时返回同一个实例（classmethod则会对基类和子类返回不同的bound method实例），但这点几乎从来没有什么时候是有用的。不过，staticmethod可以在子类上被重写为classmethod，反之亦然，因此也没有必要提前将staticmethod全部改为classmethod，按需要使用即可。

### 参考资料

[如何理解Python装饰器？](https://www.zhihu.com/question/26930016/answer/1047233982)（很详细，直接看原博即可）

[Python 中的 classmethod 和 staticmethod 有什么具体用途？](https://www.zhihu.com/question/20021164)——（暂时还是不明白二者的区别，只是感觉都相当于静态方法，只不过classmethod没有硬编码，相对来说更加方便）

## *args和**kargs的区别和使用

*是必需的，后面的args和kargs可以写成其他的名称，只是默认是这个写法。

`*args`为**可变位置参数**，传入的参数会被放进**元组**里。

`**kargs`为**可变关键字参数**，传入的参数**以键值对的形式存放到字典**里。

```python
def f(a, *args):
    print(a, args)
 
f(1,2,3,4,5)
# output------
1 (2, 3, 4, 5)
```



```python
def f(**kargs):
print(kargs)

f(a=1,b=2)
# output------
{'a': 1, 'b': 2}
```

*号的作用其实与上面的说明相反，如下例子所示：

==*将元组转换为多个单元素==

==**将字典去除key，留下的value变成多个单元素==

```python
def f1(a,b,c):
    print(a,b,c)
 
args= (1,2,3)
f1(*args)
 
def f2(a,b,c):
    print(a,b,c)
 
kargs = {
    'a':1,
    'b':2,
    'c':3
}
 
f2(**kargs)
 
1 2 3
1 2 3
```



### 参考链接

[函数中的*args 和 **kargs到底是什么东东？](https://blog.csdn.net/weixin_38754337/article/details/115410284)

## enumerate()

用在for中，将一个**可遍历的数据对象**(如列表、元组或字符串)组合为一个索引序列，同时**列出数据和数据下标**。（普通的for只返回值，而他会返回值对应的索引）

```python

>>>seq = ['one', 'two', 'three']
>>> for i, element in enumerate(seq):
...     print i, element
... 
0 one
1 two
2 three
```

## 可调用类型

**为什么可以用model(input)而不是model.forword(input)?**

可调用类型：只要类中定义了`__call___(self, param)`函数，那么该类的对象就可以当做函数来使用，且默认调用的就是`__call__()`函数。

对于pytorch中的`nn.Module`，其内部已经定义了`__call__()`函数，且其内容就是调用`self.forward()`，所以定义好模型后，使用`model(input)等同于使用model.forward()`

```python
 class A():
    def __call__(self, param):
        
        print('i can called like a function')
        print('传入参数的类型是：{}   值为： {}'.format(type(param), param))
        res = self.forward(param)
        return res
    
    def forward(self, input_):
        print('forward 函数被调用了')
        
        print('in  forward, 传入参数类型是：{}  值为: {}'.format( type(input_), input_))
        return input_
    
a = A()

input_param = a('i')
print("对象a传入的参数是：", input_param)
```

## list和np.array的区别

**List：** **列表**

python 中的 list 是 python 的内置数据类型，**list 中的数据类型不必相同**，
在 list 中保存的是**数据的存放的地址**，即指针，并非数据。



**array：数组**

array() 是 **numpy 包**中的一个函数，array 里的元素都是**同一类型**。



ndarray：

是一个多维的数组对象，具有矢量算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点。

ndarray 的一个特点是**同构**：即其中**所有元素的类型必须相同**。

NumPy 提供的 array() 函数可以将 Python 的任何序列类型转换为 ndarray 数组。

## 赋值、浅拷贝、深拷贝

- 直接赋值：其实就是对象的**引用**（别名）。
- 浅拷贝(copy)：**拷贝父对象**，不会拷贝对象的内部的子对象。
- 深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全**拷贝了父对象及其子对象**。

同C++。

浅拷贝：重新分配一块内存，创建一个新的对象，但里面的**元素是原对象中各个子对象的引用**。
只拷贝了地址，但地址指向的仍然是同一个变量。改变后者的变量值，同时也会改变原变量，造成问题。

深拷贝：重新分配一块内存，创建一个新的对象，并且将原对象中的元素，以递归的方式，通过**创建新的子对象拷贝到新对象中**。
重新开辟一块内存，并将原对象中的所有元素都拷贝到新对象中。

## 函数参数的传递形式

1. 本质上都是值传递。
2. 如果实际参数的数据类型是可变对象（列表、字典），则函数参数的传递方式将采用引用传递方式。
   引用传递方式的底层实现，采用的依然还是值传递的方式，**传递的是引用变量**，引用变量指向原对象，修改该引用变量就会修改原对象。

1. 如果需要让函数修改某些数据，则可以通过把这些数据包装成列表、字典等可变对象，然后把列表、字典等可变对象作为参数传入函数，在函数中通过列表、字典的方法修改它们，这样才能改变这些数据。

### 参考链接

[Python函数参数传递机制（超级详细）](http://c.biancheng.net/view/2258.html)

## super()用法

**super()**方法设计目的是用来解决多重继承时父类的查找问题，所以在单重继承中用不用 super 都没关系；但是，使用 super() 是一个好的习惯。一般我们**在子类中需要调用父类的方法时才会这么用**。

**super()**的好处**就是可以避免直接使用父类的名字.主要用于多重继承，如下：**

```python
class  C(P):
    def __init__(self):
            super(C,self).__init__() # 调用了父类P的构造函数
            print 'calling Cs construtor'
# 等价于
class  C(P):
     def __init__(self):
             P.__init__(self)
             print 'calling Cs construtor'
```

### 参考链接

[Python中的super()用法](https://blog.csdn.net/qq_14935437/article/details/81458506)



## import和from import的区别

import只能导入库，使用其成员时需要指明是哪个库。

而from导入时就指明了是哪个库的成员，后续使用就不用再指明是哪个库了。

## 多进程的使用方法

[](..\..\..\..\NoteBook\小程序\python\多进程使用方法.md)

## 相关文件

其他相关的记录知识点，请查看[](..\..\..\..\NoteBook\小程序\python\)
