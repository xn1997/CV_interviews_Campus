参考链接：[Shell编程(shell概述、变量操作、流程控制、函数、shell工具)](https://blog.csdn.net/weixin_44911308/article/details/109180419)

上述链接很重要

# shell 判断文件夹或文件是否存在

参考链接：[shell 判断文件夹或文件是否存在](https://www.runoob.com/note/47027)

# 命令后台执行（等待执行）

参考链接：[Linux Shell脚本后台运行查看和关闭后台运行程序的方法](http://www.itbcn.cn/?p=820)

[Bash技巧：在脚本中并发执行多个命令，等待命令执行结束](https://segmentfault.com/a/1190000021616655)

- &命令

  功能：加在一个命令的最后，可以把这个命令放在后台执行

-  `wait` 命令

  **wait [-n] [id ...]**

  `wait` 命令可以等待指定 PID 的进程执行完成。  如果不提供任何参数，则等待当前激活的所有子进程执行完成。

# 阻塞等待一定时间

- sleep命令

```shell
sleep 5s # 等待5秒 
sleep 5m # 等待5分钟
sleep 5h # 等待5小时
```



# 查看CPU信息

参考链接：[Linux查看CPU详细信息](https://www.jianshu.com/p/a0ab0ccb8051)

- 所有信息

```shell
cat /proc/cpuinfo
```

- 物理CPU的个数

```shell
cat /proc/cpuinfo | grep "physical id" | sort | uniq | wc -l
```

- 物理CPU内核个数

```shell
cat /proc/cpuinfo | grep "cpu cores" | uniq

cpu cores : 8
表示1个物理CPU里面有8个物理内核。
```

- 逻辑CPU个数

```shell
cat /proc/cpuinfo | grep "processor" | wc -l
```

- 每个物理CPU中逻辑CPU的个数

```shell
cat /proc/cpuinfo | grep "siblings" | uniq

siblings : 16
表示每个物理CPU中有16个逻辑CPU，
```

- CPU是否启用超线程

```shell
cat /proc/cpuinfo | grep -e "cpu cores" -e "siblings" | sort | uniq

cpu cores : 8
siblings : 16
看到cpu cores数量是siblings数量一半，说明启动了超线程。
如果cpu cores数量和siblings数量一致，则没有启用超线程。
```

# 相关问题

## 在shell命令行方式下，一行只能写一个指令，每次只能使用一个命令？

链接：https://www.nowcoder.com/questionTerminal/5757dd82700c4e6194bd0022337c0e8a?orderByHotValue=1&mutiTagIds=239&page=1&onlyReference=false

shell中可以通过一行执行多个命令。有以下三种方式： 

  1、多个命令通过**分号；隔离**，表示所分隔的命令会连续的执行下去，**就算是错误的命令也会继续执行后面的命令** 

  2、多个命令通过**&&隔离**，表示命令也会一直执行下去，但是中**间有错误的命令存在就不会执行后面的命令**，没错就直行至完为止。 

  3、多个命令通过**||隔离**，表示一**遇到可以执行成功的命令就会停止执行后面的命令**，而不管后面的命令是否正确与否。如果执行到错误的命令就是继续执行后一个命令，一直执行到遇到正确的命令为止