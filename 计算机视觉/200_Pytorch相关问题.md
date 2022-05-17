## 模型搭建流程

**模型搭建流程**

1. 加载数据。`train_loader = torch.utils.data.Dataloader()`
2. 构建模型。`model = torch.nn.module`
3. 构建损失。`loss = torch.nn.CrossEntropyLoss()`
4. 构建优化器。`optimizer = torch.optim.Adam()`
5. 设置学习率更新策略。`scheduler = torch.optim.lr_scheduler.MultiStepLR()`
6. FP16，设置FP16更新器。`scaler = torch.cuda.amp.GradScaler()`

**更新流程**

1. 读取数据。`for n_iter, (img, label) in enumerate(train_loader)`
2. 前向传播。`output = model(img)`
3. loss计算。`loss(output, label)`
4. 反向传播，计算梯度。`loss.backward() or scaler.scale(loss).backward()`
5. 如果有FP16，梯度还原。`scaler.unscale_(optimizer)`
6. 使用优化器由梯度更新权重。`optimizer.step() or scaler.step(optimizer)`
7. 整个epoch结束后，更新学习率。`scheduler.step()`

详细流程参考：[xn1997/naic2020_re-id](https://gitee.com/xn1997/naic2020_re-id/blob/master/NAIC_Person_ReID_DMT/NAIC_Person_ReID_DMT/processor/processor_my.py#L84)

## function和module有什么区别

1）Function一般只定义一个操作，因为其无法保存参数，因此适用于激活函数、pooling等操作；Module是保存了参数，因此适合于定义一层，如线性层，卷积层，也适用于定义一个网络。

2）Function需要定义三个方法：__init__, forward, **backward（需要自己写求导公式）；Module：只需定义__init__和forward，而backward的计算由自动求导机制构成。**

3）可以不严谨的认为，Module是由一系列Function组成，因此其在forward的过程中，Function和Variable组成了计算图，在backward时，只需调用Function的backward就得到结果，因此Module不需要再定义backward。

4）Module不仅包括了Function，还包括了对应的参数，以及其他函数与变量，这是Function所不具备的

### 参考链接

https://www.jianshu.com/p/33753873911a

## dataloader、dataset、sampler有什么区别

PyTorch中Dataset, DataLoader, Sampler的关系可用下图概括：

<img src="https://img2020.cnblogs.com/blog/1800705/202005/1800705-20200508230314846-581250372.png" alt="img" style="zoom:50%;" />

用文字表达就是：

Dateloader中包含Sampler和Dataset，使用sampler产生索引，然后通过索引去Dataset中找到对应的数据。

Sampler产生索引，返回的也就是一批索引值。

Dataset拿着这个索引在数据集文件夹中找到对应的样本（每个样本对应一个索引，就像列表中每个元素对应一个索引），及标签，并返回（样本+标签）给调用方。

在`enumerate(Dateloader对象)`过程中，Dataloader按照其参数BatchSampler规定的策略调用其Dataset的getitem方法batchsize次，得到一个batch，该batch中既包含样本，也包含相应的标签。

- 一般只需要定义Dataset即可，shuffle=True（随机读取数据），batch_size=（一次读取的数据数），num_workers=（读取一个batch数据时所使用的进程数，并行读取数据，速度快）

- Dataset主要重写`__getiterm__`，以便根据sample得到的索引来获取数据和标签

### 参考链接

[PyTorch中Dataset, DataLoader, Sampler的关系](https://www.cnblogs.com/picassooo/p/12853672.html)（简要概括，同上总结）

[一文弄懂Pytorch的DataLoader, DataSet, Sampler之间的关系](https://www.cnblogs.com/marsggbo/p/11308889.html)（详细介绍）

[PyTorch源码解析与实践（1）：数据加载Dataset，Sampler与DataLoader](https://zhuanlan.zhihu.com/p/270028097)——介绍的更形象具体一些。

### DataLoader API解析

- **num_workers** (int, optional)： **提高数据的加载速度（多进程）**
  这个参数决定了有**几个进程来处理**data loading。0意味着所有的数据都会被load进主进程。（默认为0）
- **pin_memory** (bool, optional)：
  如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中。**提高数据从cpu到gpu传输效率**。
- drop_last (bool, optional):
  如果设置为True：**最后一个batch不足设置的batch大小时就直接丢弃。**这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了…
  如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。

#### pin_memory

**1.锁页内存与不锁页内存**

主机中的内存根据是否与虚拟内存(虚拟内存指的是硬盘)进行交换分为两种，一种是锁页内存，另一种是不锁页内存。

- 锁页内存指存放的内容在任何情况下都不与主机虚拟内存进行交换。
- 不锁页内存指在主机内存不足时，数据存放在虚拟内存中。

**显卡中的显存全部是锁页内存！**

**2.pin_memory**

创建Dataloader时，pin_memory=True表示将load进的数据拷贝进锁页内存区，将内存中的Tensor转移至GPU cuda区会很快；pin_memory=False表示将load进数据放至硬盘，速度会较慢。

当计算机的内存充足的时候，设置pin_memory=True。当系统卡住，或者交换内存使用过多的时候，设置pin_memory=False。因为pin_memory与电脑硬件性能有关，pytorch开发者不能确保每一个炼丹玩家都有高端设备，因此pin_memory默认为False。

### num_workers

1. 每每轮到dataloader加载数据时：

   ```python
   for epoch in range(start_epoch, end_epoch):
       for i, data in enumerate(trainloader):
   ```

   dataloader一次性创建`num_worker`个worker，（也可以说dataloader一次性创建`num_worker`个工作进程，worker也是普通的工作进程），

   并用`batch_sampler`将指定batch分配给指定worker，worker将它负责的batch加载进RAM。

   然后，dataloader从RAM中找本轮迭代要用的batch，如果找到了，就使用。如果没找到，就要`num_worker`个worker继续加载batch到内存，直到dataloader在RAM中找到目标batch。一般情况下都是能找到的，因为`batch_sampler`指定batch时当然优先指定本轮要用的batch。

2. `num_worker`设置得大，**好处是寻batch速度快**，因为下一轮迭代的batch很可能在上一轮/上上一轮...迭代时已经加载好了。**坏处是内存开销大**，也加重了CPU负担（worker加载数据到RAM的进程是CPU复制的嘛）。`num_workers`的经验设置值是自己电脑/服务器的CPU核心数，如果CPU很强、RAM也很充足，就可以设置得更大些。

3. 如果`num_worker`设为0，意味着每一轮迭代时，dataloader不再有自主加载数据到RAM这一步骤（因为没有worker了），而是在RAM中找batch，找不到时再加载相应的batch。缺点当然是速度更慢。

## Tensor及numpy通道转换

numpy的通道转换：**transpose**

```python
img.transpose([2,0,1])
```

pytorch(tensor)的通道转换：**permute**

```python
img.permute(2,0,1)
# 例2
>>> torch.randn(2,3,4,5).permute(3,2,0,1).shape
torch.Size([5, 4, 2, 3])
```

**permute相当于可以同时操作于tensor的若干维度，transpose只能同时作用于tensor的两个维度；**

==view和reshape功能一样，但是他们和permute不同，view是按行优先的顺序展平tensor，然后再填充到指定维度，而permute直接是维度的转换，他们最终的数据是不一样的==

**补充**：

`tensor.contiguous().view()`等价于`torch.reshape()`（等于`numpy.reshape()`）

**contiguous**：view只能作用在contiguous的variable上，如果在view之前调用了transpose、permute等，就需要调用contiguous()来返回一个contiguous copy；

一种可能的解释是：有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的**view()操作依赖于内存是整块的**，这时只**需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式**；

### 参考链接

[Pytorch之permute函数](https://zhuanlan.zhihu.com/p/76583143)

## Tensor与numpy/PIL的转换

```python
# numpy 2 tensor
tensor = torch.from_numpy(ndarray)
# tensor 2 numpy
img.cpu().numpy()

# list 2 tensor
torch.Tensor(list)
# tensor 2 list
list.numpy().tolist()

# ------与PIL的转换------
# PIL 2 tensor
from torchvision import transforms as T

img = Image.open(img_path).convert('RGB')
image = transforms.ToTensor()(image)
	# transforms一般使用方法为：
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN),  # 调整到设定大小
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),  # 水平反转
        # T.Pad(cfg.INPUT.PADDING),
        # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        RandomShiftingAugmentation(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),  # 正则化
        RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)  # 随机擦除
    ])

# tensor 2 PIL
image = transforms.ToPILImage()(image)
```

transforms使用参考：[NAIC reid比赛代码](https://gitee.com/xn1997/naic2020_re-id/blob/master/NAIC_Person_ReID_DMT/NAIC_Person_ReID_DMT/datasets/make_dataloader.py)

## 常用函数

Tensor的维度表示：N，C，H，W

### clamp（截断函数）

```python
torch.clamp(input, min, max, out=None) → Tensor
# 或者
x = x.clamp(min=,max=)
```

将所有张量截断到min，max之间

### cat（拼接）

```python
combine = torch.cat([d1, add1, add2, add3, add4], dim=1)
```

dim：在第几个维度进行拼接。从0开始

### model.train和model.eval区别

主要是针对BN和Dropout这种在训练和预测期间操作不同的结构。

**train阶段：**BN和dropout在训练中起到防止过拟合的作用。

**eval阶段：**①BN的参数直接固定，不会再被改变，BN不会再计算输入数据的均值方差，而是直接使用训练集统计出的均值方差，这样就可以避免test的batchsize过小，使得计算出的均值方差不具有统计特征，使结果非常的差；②使Dropout不起作用：训练时随机失活，推理时全部开启，同时最终输出要乘以失活比例，否则会导致最后的结果翻倍。

torch.no_grad()和expend和unquee

## ModuleList和Sequential的区别

nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。

1. nn.Sequential内部实现了forward函数，因此可以不用写forward函数。而nn.ModuleList则没有实现内部forward函数。
2. nn.Sequential中可以使用OrderedDict来指定每个module的名字。
3. nn.Sequential里面的模块按照顺序进行排列的，所以必须确保前一个模块的输出大小和下一个模块的输入大小是一致的。而nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起，这些模块之间并没有什么先后顺序可言，执行顺序由forward函数决定。
4. 有的时候网络中有很多相似或者重复的层，我们一般会考虑用 for 循环来创建它们(nn.ModuleList)，而不是一行一行地写(nn.Sequential)。

一般情况下 nn.Sequential 的用法是来组成卷积块 (block)，然后像拼积木一样把不同的 block 拼成整个网络，让代码更简洁，更加结构化。而对于重复的层，就使用nn.ModuleList及for来创建。

### nn.ModuleList

不同于一般的 list，**加入到 nn.ModuleList 里面的 module 是会自动注册到整个网络上**的，同时 module 的 parameters 也会自动添加到整个网络中。若使用python的list，则会出问题。使用 Python 的 list 添加的卷积层和它们的 parameters 并没有自动注册到我们的网络中。当然，我们还是可以使用 forward 来计算输出结果。但是如果用其实例化的网络进行训练的时候，因为这些层的parameters不在整个网络之中，所以其网络参数也不会被更新，也就是无法训练。

```python
"""nn.ModuleList使用例子"""
class net4(nn.Module):
    def __init__(self):
        super(net4, self).__init__()
        layers = [nn.Linear(10, 10) for i in range(5)] # ++++++++
        self.linears = nn.ModuleList(layers) # ++++++++

    def forward(self, x):
        for layer in self.linears:
            x = layer(x)
        return x

net = net4()
print(net)
# net4(
#   (linears): ModuleList(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): Linear(in_features=10, out_features=10, bias=True)
#     (2): Linear(in_features=10, out_features=10, bias=True)
#   )
# )
```

### 参考链接

[详解PyTorch中的ModuleList和Sequential](https://zhuanlan.zhihu.com/p/75206669)

## 训练时GPU显存不足的处理方法

参考链接：[深度神经网络模型训练时GPU显存不足怎么办？](https://blog.csdn.net/Zserendipity/article/details/105301983)

### Relu 的 inplace 参数设为True

激活函数 Relu() 有一个默认参数 inplace ，默认为Flase， 当设置为True的时候，我们在通过relu() 计算得到的新值不会占用新的空间而是直接覆盖原来的值，这表示设为True， 可以节省一部分显存。

### **梯度累计**

**batch无论多大，loss都一样，因为loss和batch无关，所以当batch大了，必须提高学习率，确保一个epoch内的更新幅度一致**

**（实现GPU不足情况下，依然可以实现大batch的训练，非常实用）**

传统训练：

```python
for i,(feature,target) in enumerate(train_loader):
    outputs = model(feature)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
 
    optimizer.zero_grad()   # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 反向传播， 更新网络参数
```

加入梯度累加之后：

```python
for i,(features,target) in enumerate(train_loader):
    outputs = model(images)  # 前向传播
    loss = criterion(outputs,target)  # 计算损失
    loss = loss/accumulation_steps   # 一般需要除以组数，因为loss一般都做了归一化，所以无论batch多大，最终得到的损失都一样，所以为了保证最终大batch的loss不变，就必须除以次数
 
    loss.backward()  # 计算梯度
    if((i+1)%accumulation_steps)==0:
        optimizer.step()        # 反向传播，更新网络参数
        optimizer.zero_grad()   # 清空梯度
```

更详细来说， 假设 batch size = 32， accumulation steps = 8 ， 梯度积累首先在前向传播的时候讲  batch 分为 accumulation steps 份， 然后得到 size=4 的小份batch ， 每次就以小 batch  来计算梯度，但是不更新参数，将梯度积累下来，直到我们计算了 accumulation steps 个小 batch， 我们再更新参数。

# 自己学习使用

## 爱因斯坦求和函数(enisum)

参考链接：[【深度学习中的神仙操作】einsum爱因斯坦求和](https://zhuanlan.zhihu.com/p/74462893)——非常详细，例子易理解

## 参数初始化

常见的初始化权重写法

```python
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)
model.apply(weights_init) # 模型权值初始化
```

###### 分类

`xavier_normal_`初始化：针对激活函数为sigmoid的

`kaiming_normal_`初始化：针对激活函数为relu的

`constant_`：初始化为常数



## 设置随机种子，使训练结果可复现

参考链接：[Pytorch设置随机数种子，使训练结果可复现。](https://zhuanlan.zhihu.com/p/76472385)

```python
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
# 预处理数据以及训练模型
# ...
# ...
```



## torch.nn.MSELoss(reduction='mean')

计算偏差的**平方和均值**，因此通常计算loss时，可以**手动乘以1/2**以确保反向传播求导容易些

- 如果不是`'mean'`，那返回的就是平方和
- 如果是`'none'`，返回的就是一个数组，包含了所有位置的**偏差平方**

## hook函数

搭配`model[layer_num].register_forward_hook(hook)` 函数。

`model[layer_num]`就是网络结构，内容形如下表

```yaml
Sequential(
  (0): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU6(inplace=True)
  )
  (1): InvertedResidual(
    (conv): Sequential(
      (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU6(inplace=True)
      (3): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
      (4): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
)
```

- 为`model[layer_num]`层添加一个`hook(钩子)`函数，一旦该层进行了`foward(推理)`，`hook`函数就会被调用。

- `hook`不会改变网络的`input和output`，不会对网络产生任何影响，一般用来打印网络参数使用。

```python
def hook(module,input,output):
   class_name = str(module.__class__.__name__)
   if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
           class_name.find("Linear") != -1:
       params = 0
       for param_ in module.parameters():
           params += param_.view(-1).size(0)
       print('params',params)

import torch
conv = torch.nn.Conv2d(1,8,(2,3))
input = torch.rand(1,1,224,224) # batch,channel,width,height
hook_handle = conv.register_forward_hook(hook)
output = conv(input)
hook_handle.remove()
output = conv(input)
```

## torch.gather

参考链接：[Pytorch系列（1）：torch.gather()](https://blog.csdn.net/cpluss/article/details/90260550)

```python
torch.gather(input, dims=1, index = LongTensor(或者numpy))
```

- 利用`index`来索引`input`-第`dims`维度-指定位置的数值
- 返回值维度和`index`一模一样



## torch.nn.DataParallel无法导出到onnx

应该去掉torch.nn.DataParallel属性，具体操作方法参考链接[Onnx EfficientNet网络转onnx格式出现的问题记录](https://blog.csdn.net/qq_33120609/article/details/96998546)

即自己设置一个有序字典，有选择的读取模型数据，然后保存成torch.onnx可以读取的格式，即可，代码为：

```python
from collections import OrderedDict
state_dict=torch.load(model_path)
new_state_dict=OrderedDict()
for k,v in state_dict.items():
	name=k[7:] # 也就是去掉最开头的关键字’modules.’
	new_state_dict[name]=v
net.load_state_dict(new_state_dict)
```

