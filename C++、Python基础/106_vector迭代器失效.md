## 迭代器失效情况

在STL中，迭代器失效可发生在三种情况下：

### 一、数组型数据结构（vector、deque）

对于序列式容器(如vector,deque)，序列式容器就是数组式容器

1. **删除当前的iterator或者插入某个iterator会使后面所有元素的iterator都失效**。这是因为vetor,deque使用了连续分配的内存，删除或插入一个元素导致后面所有的元素会向前或向后移动一个位置。所以不能使用erase(iter++)的方式，**还好erase，insert方法可以返回下一个有效的iterator，以此解决无效迭代器的问题**。
2. 当使用push_back()，则当容器不重新分配，即size小于capacity时，指向容器结尾的迭代器，即插入前计算得到的v.end(）返回的迭代器会失效。**当内存重新分配后，这原先的所有迭代器都会失效**。

**解决方法：**(所有迭代器失效都用该方法解决)
（1）通过erase方法的返回值来获取下一个有效的迭代器，如下例。
（2）在调用erase之前，先使用‘++’来获取下一个有效的迭代器

```cpp
vector<int> cont;
for (auto iter = cont.begin(); iter != cont.end();)
{
   (*it)->doSomething();
   if (shouldDelete(*iter))
      iter = cont.erase(iter);  //erase删除元素，返回下一个迭代器
   else
      ++iter;
}
```

### 二、链表型数据结构（list）

使用了不连续分配的内存，删除运算使指向删除位置的迭代器失效，但是不会失效其他迭代器。还好erase，insert方法可以返回下一个有效的iterator。
解决方法：
通过erase方法的返回值来获取下一个有效的迭代器，做法和序列式容器完全相同。

```cpp
for (iter = cont.begin(); it != cont.end();)
{
   (*iter)->doSomething();
   if (shouldDelete(*iter))
      cont.erase(iter++);
   else
      ++iter;
}
```



### 三、树形数据结构（map、set、multimap,multiset）

删除当前的iterator，仅仅会使当前的iterator失效，只要在erase时，递增当前iterator即可。这是因为map之类的容器，使用了红黑树来实现，插入、删除一个结点不会对其他结点造成影响。erase迭代器只是被删元素的迭代器失效，但是返回值为void，所以要采用erase(iter++)的方式删除迭代器。

解决方法：
（1）采用erase(iter++)的方式删除迭代器。如下第一个例子：
（2）在调用erase之前，先使用‘++’来获取下一个有效的迭代器。如下第二个例子。

```cpp
map<int,int> cont;
for (auto iter = cont.begin(); iter != cont.end();)
{
   (*it)->doSomething();
   if (shouldDelete(*iter))
      iter = cont.erase(iter++);  //erase删除元素，返回下一个迭代器
   else
      ++iter;
}
```



```cpp
for (iter = dataMap.begin(); iter != dataMap.end(); )
{
    int nKey = iter->first;
    string strValue = iter->second;
    if (nKey % 2 == 0)
    {
        map<int, string>::iterator tmpIter = iter;
        iter++;
        dataMap.erase(tmpIter);
        //dataMap.erase(iter++) 这样也行
 
    }else
 	{
      iter++;
    }	
}
```
## 总结

迭代器失效分三种情况考虑，也是分三种数据结构考虑，分别为数组型，链表型，树型数据结构。

**数组型数据结构：**该数据结构的元素是分配在连续的内存中，insert和erase操作，都会使得删除点和插入点之后的元素挪位置，所以，==插入点和删除掉之后的迭代器全部失效==，也就是说`insert(*iter)(或erase(*iter))`，然后在iter++，是没有意义的。解决方法：`erase(*iter)`的返回值是下一个有效迭代器的值。 `iter =cont.erase(iter)`;

**链表型数据结构：**对于list型的数据结构，使用了不连续分配的内存，删除运算使指向==删除位置的迭代器失效，但是不会失效其他迭代器==.解决办法两种，`erase(*iter)`会返回下一个有效迭代器的值，或者`erase(iter++)`.

**树形数据结构：** 使用红黑树来存储数据，==插入不会使得任何迭代器失效==；==删除运算使指向删除位置的迭代器失效，但是不会失效其他迭代器==.erase迭代器只是被删元素的迭代器失效，但是**返回值为void，所以要采用erase(iter++)的方式删除迭代器**。

==注意：经过erase(iter)之后的迭代器完全失效，该迭代器iter不能参与任何运算，包括`iter++,*ite`==

## 参考链接

[C++迭代器失效的情况与解决方法](https://blog.csdn.net/weixin_42579072/article/details/107568814)（对下面链接的一个总结，比较清晰易读）

[C++迭代器失效的几种情况总结](https://www.cnblogs.com/fnlingnzb-learner/p/9300073.html)（特别详细，上面链接不理解的直接看该链接）