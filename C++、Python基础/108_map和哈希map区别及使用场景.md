## 哈希map(unordered_map)

在 `unordered_map` 内部，==使用的 **`Hash Table`** 对数据进行组织==，通过把键值 `key` 映射到 `hash` 表中的一个位置进行访问，根据 `hash` 函数的特点， `unordered_map` 对于**元素查找的时间复杂度可以达到 `O(1)`** ，但是，它的**元素排列是无序的**。具体例子如下：

```cpp
int main() {
    using namespace std;
    // 首先创建一个无序 map，它的 key 使用 int 类型，value 使用 string 类型
    unordered_map<int, string> unorderedMap;    
   
    // 三种插入新元素的方法，“茴”字有三种写法~
    unorderedMap.insert(make_pair(0, "Alice")); 
    unorderedMap[1] = "Bob";
    unorderedMap.insert(unordered_map<int, string>::value_type(2, "Candy"));
 
    // 对内部元素挨个输出
    for (auto iter = unorderedMap.begin(); iter != unorderedMap.end(); iter++) {
        cout << iter->first << " - " << iter->second << endl;
        /*
         * >: 输出如下，可以得知它们在 key 的排序上并没有顺序
         * 2 - Candy
         * 0 - Alice
         * 1 - Bob
         */
    }
}
```

==`unordered_map` 由于建立了哈希表，所以它在最开始建立的时候比较耗时间，但是它查询速度快==，一般情况下用 `unordered_map` 是没有问题的。

## map

在 `map` 的内部，==使用了**「红黑树」**（`red-black tree`）来组织数据==，因此默认的就已经实现了数据的排序。从下面例子中可以看出，它==默认实现了在 `key` 上排序==实现递增：

```cpp
int main() {
    map<int, string> mapper;
    mapper.insert(make_pair(0, "Alice"));
    mapper[1] = "Bob";
    mapper.insert(map<int, string>::value_type(2, "Candy"));
    for (auto &iter : mapper) {
        cout << iter.first << " - " << iter.second << endl;
        /*
         * >: 输出如下，很明显的，它们在 key 的排序上是递增排列的
         * 0 - Alice
         * 1 - Bob
         * 2 - Candy
         */
    }
}
```

不过，==在存储上 `map` 却比较占用空间，因为在红黑树中，每一个节点都要额外保存父节点和子节点的连接，因此使得每一个节点都占用较大空间来维护红黑树性质==。

## **优缺点以及适用处**

两种数据结构特点如下表格

![img](https://pic1.zhimg.com/80/v2-e7a0e37ae8457359287f99f9859dda28_720w.jpg)

**unordered_map：**

**优点：** 因为内部实现了哈希表，因此其==查找速度非常的快==
**缺点：** ==哈希表的建立比较耗费时间==
**适用处：**对于查找问题，unordered_map会更加高效一些，因此遇到==查找问题，常会考虑一下用unordered_map==

**map：**

**优点：**==有序性==，这是map结构最大的优点，其元素的有序性在很多应用中都会简化很多的操作。红黑树，内部实现一个红黑书==使得map的很多操作在$lg(n)$的时间复杂度下就可以实现，因此效率非常的高==
**缺点：**空间占用率高，因为map内部实现了红黑树，虽然提高了运行效率，但是因为每一个节点都需要额外保存父节点、孩子节点和红/黑性质，使得==每一个节点都占用大量的空间==

**适用处：**对于那些有顺序要求的问题，用map会更高效一些

<big>**注意：**</big>

内存占有率的问题就转化成红黑树 VS hash表 , 还是==unordered_map占用的内存要高。==
但是==unordered_map执行效率要比map高很多==
对于unordered_map或unordered_set容器，**其遍历顺序与创建该容器时输入的==顺序不一定相同==，因为遍历是按照哈希表从前往后依次遍历的**

## 参考链接

[c++ map与unordered_map区别及使用](https://blog.csdn.net/qq_21997625/article/details/84672775)（这个比较详细，建议主要看这个）

[map和unordered_map的区别](https://zhuanlan.zhihu.com/p/210458185)

## 哈希map如何使用自定义类作为key

哈希map的实现需要两个条件：

- 可以判断两个class变量是否相等的函数。（直接重载operator==即可）
- 可以计算class地址的函数。（实现一个std::hash模板类，重载operator()）

```cpp
class Myclass{
public:
    int first;
    vector<int> second;
    // 重载判断class是否相等的函数
    bool operator==(const Myclass &other) const{
        return first == other.first && second == other.second;
    }
}
// 实现Myclass的hash函数
namespace std{
    template <> // ???暂时不知道有何用，建议实际尝试后再决定是否保留该句
    struct hash<Myclass>{
        size_t operator()(const Myclass &k) const{
            // 自己设计地址计算方法即可（随便写，尽量使不同的变量有不同的返回值）
            int res = k.first;
            for(auto x : k.second){
                res = res ^ x;
            }
            return res;
        }
    };
}
int main(){
    unordered_map<Myclass, double> s;
    Myclass a = { 2, {3, 4} };
    Myclass b = { 3, {1, 2, 3, 4} };
    S[a] = 2.5;
    S[b] = 3.123;
    cout << S[a] << ' ' << S[b] << endl;
    return 0;
}
```

### 参考链接

[C++ 之 unordered_map——哈希表](https://zhuanlan.zhihu.com/p/339356935)

