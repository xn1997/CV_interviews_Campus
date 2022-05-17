# Git使用方法

<big><font color=red>个人仓库建议使用gitee，上传下载速度非常快，使用方法同github一模一样</font></big>

参考链接：[如何配置环境](https://blog.csdn.net/ZY1732331520/article/details/108410464)

[如何管理库](https://blog.51cto.com/dorebmoon/2334354)

完整的学习教程：[菜鸟教程](https://www.runoob.com/git/git-workflow.html)

## 配置个人信息

随便起名字，随便给个邮箱，不重要

```shell
git config --global user.name “xn1997”
git config --global user.email “759700567@qq.com”
```

`git config`：用于配置或读取相应的工作环境变量

选项包括：

1. `--system`：读写系统配置，即读取的`/etc/gitconfig`文件
2. `--global`：读写该用户的配置，读取的`~/.gitconfig`文件

如果不使用任何选项：就是读取的当前项目的`.git/config`文件，即只在当前项目中使用这些配置。

`git config --list`：显示当前的git 配置信息。

## 生成密钥

邮箱同上一步一样

```shell
ssh-keygen -t rsa -C “759700567@qq.com”
```

一路回车，会在用户主目录`.ssh/id_rsa.pub `生成密钥文件，打开后复制。

```shell
gedit ~/.ssh/id_rsa.pub
```

## 登记密钥

登录网页github

选择`Settings->SSH and GPG keys->NEW SSH key`

粘贴后并设置标题，ADD SSH key



到此环境配置完成，就可以使用git命令类

## 使用方法

### 常用API

1. 在网页上创建一个库，如`NAIC2020`，默认分支设定为`master`

2. 拷贝仓库到本地

   ```shell
   git clone https://github.com/xn1997/NAIC2020.git
   cd cd NAIC2020/  # 必须进入仓库目录进行操作
   ```

3. 直接再本地`NAIC2020/`文件内修改即可

4. 把文件添加到仓库

   ```shell
   git add .  # 添加所有的文件到仓库，也可指定文件
   ```

5. 把文件提交到仓库(暂存区--->仓库)

   ```shell
   git commit -m "对该版本做一点描述即可"
   ```

6.  commit 的消息内容填错再修改

   ```shell
   git commit --amend
   ```

   

7. 把本地库上传到github里

   ```shell
   git push <远程主机名> <本地分支名>
   ```

   以下，将本地`master`分支推送到`origin`主机的`master`分支*（origin是远程链接.git的别名）*

   ```shell
   git push -u origin master
   ```

   只用`git push`也可以上传

   如果本地和远程有差异，强制推送使用`--force`参数

8. 查看当前的文件修改、所处分支等等....

   ```shell
   git status
   ```

9. 比较暂存区和工作区的差异

   ```shell
   git diff
   ```

10. 撤销`git add .`操作

   `git reset`：版本回退

   ```shell
   git reset HEAD .  # 撤销所有操作
   git reset HEAD 文件名  # 撤销单个文件
   git reset HEAD^ # 回退所有内容到上一个版本
   git reset 052e # 回退到指定版本
   git reset --soft HEAD~n # 回退到n前个版本
   git reset --hard HEAD # 将暂存区和工作区都回到上一个版本，并删除所有的信息提交（一般不用）
   ```

   **软撤销 --soft**

   本地代码不会变化，只是 git 转改会恢复为 commit 之前的状态

   ==不删除工作空间改动代码，撤销 commit，不撤销 git add .==

   ```bash
   git reset --soft HEAD~1
   ```

   表示撤销最后一次的 commit ，1 可以换成其他更早的数字

   **硬撤销 --hard**

   本地代码会直接变更为指定的提交版本，**慎用**

   ==删除工作空间改动代码，撤销 commit，撤销 git add .==

   注意完成这个操作后，就恢复到了上一次的commit状态。

   ```bash
   git reset --hard HEAD~1
   ```

11. 删除远程仓库文件

   直接修改.gitignore文件,将不需要的文件过滤掉，然后执行命令:

   ```shell 
git rm -r --cached .  # 删除本地缓存
git add .  # 依据.gitignore重新生成缓存
git commit
git push  -u origin master
   ```

11. 删除本地与远程仓库的链接

    ```shell
    git remote rm origin # origin是该远程链接的别名
    ```

    

12. `git rm`

    1. 将文件从暂存区和工作区删除

    2. 如果修改过该文件且存放到了暂存区，应该使用强制删除选项-f

       ```shell
       git rm -f a.txt
       ```

    3. 把文件从暂存区删除，但仍保留在当前工作区。即只从跟踪清单中删除，使用`--cached`

       ```shell
       git rm --cached a.txt
       ```

    4. 如果是删除的目录，需要使用`-r`


### git commit后如何撤销commit

参考上面的`get reset`指令

> --soft
> 不删除工作空间的改动代码 ，撤销commit，不撤销git add file

> --hard
> 删除工作空间的改动代码，撤销commit且撤销add

如果commit注释写错了，先要改一下注释，有其他方法也能实现，如：

> git commit --amend
> 这时候会进入vim编辑器，修改完成你要的注释后保存即可。


### 修改本地链接到的远程链接地址

1. 查看当前链接的远程仓库地址

   ```shell
   git remote -v
   ```

2. 修改链接地址

   ```shell
   git remote set-url origin https://github.com/*****.git
   # 直接运行git push即可上传
   ```


### 直接将本地文件夹同步成远程的仓库

1. 先在远程建立好一个空仓库

2. 在本地文件夹下执行

   ```shell
   git init # 使用当前目标初始化仓库（也可跟文件夹，指定目录生成Git仓库）
   git remote add origin https://gitee.com/xn1997/TensorRT.git
   # 然后使用git add . -> git commit ->
git push --set-upstream origin master # 完成提交
   ```
   
   

### 分支管理

1. 创建分支

   ```shell
   git branch dev  # 创建分支dev
   ```

2. 切换分支

   ```shell
   git checkout dev # 切换为本地分支
   git checkout -b 本地分支名(自己随意取) origin/远程分支名 # 拉取远程分支，同时创建对应的本地分支，并切换到该分支
   ```

3. 查看所有分支

   ```shell
   git branch # 查看本地分支
   git branch -r # 查看远程分支
   git branch -a # 查看本地+远程分支
   ```

4. 查看当前状态（所在分支、文件改动）

   ```shell
   git status
   ```

5. 删除分支

   ```shell
   git branch -d dev
   ```

6. 合并分支

   ```shell
   git merge dev
   ```

   将dev分支合并到master中

### 打Git标签

重点记住某一个版本，类似别人下载某个版本

```shell
git tag -a v1.0
```

`-a`：会让提交对该版本的注解，同提交注解

- 删除标签

  ```shell
  git tag -d v1.0
  ```

  删除远程标签

  ```shell
  git push origin :refs/tags/tagname
  ```

  

- 查看已有标签

  ```shell
  git tag
  ```

- 查看该标签版本所有的修改内容

  ```
  git show v1.0
  ```

- 此时远程仓库并不会出现该标签，需要显示的上传该标签到远程仓库

  ```shell
  git push origin v1.0
  ```

  

### 忽略配置文件(gitignore)编写

```shell
touch .gitignore
```

在`.gitignore`文件内写入想要忽略的文件

```yaml
*.mp4  # 去除所有mp4格式的文件
!test.mp4  # test.mp4不忽略
build/  # 忽略整个build文件夹 

# 忽略build文件夹下，除a.txt的所有文件
build/*  # 必须加*，否则下面那个不忽略的指令无法使用
!build/a.txt
```

* 多级文件夹忽略（如忽略`main/build`）

  在`main`文件夹下建立`.gitignore`文件，写入`build/`即可

**可执行文件的忽略**

```yaml
# 如果可执行文件名为elevator_monitor，直接写elevator_monitor，会将其视为文件名，无法忽略，应该写为下面的语句
*levator_monitor
```

#### 对于已经上传过的文件进行忽略

```shell
git rm --cached <file> # 删除文件的追踪状态
or
git rm -r --cached <path> # 删除文件夹的追踪状态
# 修改.gitignore（什么时候修改都可以）
# add+commit+push完成提交
```

**原理：**

1. `.gitignore`文件**本质是忽略未追踪文件(untracked files)**，也就是那些从未被git记录过的文件（自添加以后，从未add及commit过的文件）。因此如果某个文件被Git记录过，那么即便修改`.gitignore`对这些文件也是无效的。
2. `git rm --cached <file>`**删除的是文件的追踪状态**，而不是物理文件。因此，删除文件的追踪状态后，该文件也就没有被Git记录过了，`.gitignore`也就生效了。

#### 忽略当前文件及子文件内的内容

比如去除Python中的`__pycache__`文件夹

```shell
git rm --cached */__pycache__/* # 删除所有__pycache__的追踪状态
# .gitignore中写入
**/__pycache__ # 这样才能连同子文件夹内的__pycache__文件也删除
# 如果写入
*/__pycache__ # 这样只会删除一级文件夹内的__pycache__，子文件夹内的无法删除
```



### 文件冲突的解决

```shell
git status # 查看文件状态，查看是否有未提交的文件
git stash # 暂存修改，工作区恢复至上一次提交的结果
git pull # 拉取最新的代码
git stash pop # 合并暂存的修改到工作区，此时如果有冲突会提示，然后手动解决冲突即可
```



## 工作区、暂存区、版本库

- **工作区：**就是你在电脑里**能看到的目录**。
- **暂存区：**英文叫 stage 或 index。一般存放在  .git 目录下的 index 文件（**`.git/index`**）中，所以我们把暂存区有时也叫作索引（index）。
- **版本库：**工作区有一个**隐藏目录 .git**，这个不算工作区，而是 Git 的版本库。

`objects`：Git 的**对象库**，位于`.git/objects` 目录下，里面包含了创建的各种对象及内容

`git add `：暂存区的目录树被更新，同时工作区修改（或新增）的文件内容被写入到对象库中的一个新的对象中，而该对象的ID被记录在暂存区的文件索引中

`git commit`：暂存区的目录树写到版本库（对象库）中，**master 分支会做相应的更新**

` git reset HEAD `：暂存区的目录树会被重写，被 master 分支指向的目录树所替换，但是工作区不受影响。

`git rm --cached <file> `：会直接**从暂存区删除文件**，工作区则不做出改变。

` git checkout . 或者 git checkout -- <file>` ：会用暂存区全部或指定的文件**替换工作区的文件**。这个操作很危险，会清除工作区中未添加到暂存区的改动。

`git checkout HEAD . 或者 git checkout HEAD <file> `：会用 HEAD 指向的 master 分支中的全部或者部分文件**替换暂存区和以及工作区中的文件**。这个命令也是极具危险性的，因为不但会清除工作区中未提交的改动，也会清除暂存区中未提交的改动。