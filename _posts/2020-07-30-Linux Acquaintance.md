---
title: Linux的初认识
layout: post
categories: Linux
tags: 综述
excerpt: 对Linux系统的初步了解整理
---

# 目录 <span id="home">

* **[1. 引言](#1)**
* **[2. 主要内容](#2)**
  * **[2.1 Linux系统](#2.1)**
  * **[2.2 ](#2.2)**
  * **[2.3 ](#2.3)**
* **[3. 总结](#3)**
* **[4. 参考列表](#4)**

# 1. 引言 <span id="1">  

在IT领域，必不可少的需要对Linux系统具有实践操作和编程的能力。因此，整理这篇博文的目的就是整理一个入门Linux的初见解。

# 2. 主要内容<span id="2">  

对Linux的认识，从系统操作和编程使用两个方面去整理归纳。

#### 2.1 Linux系统<span id='2.1'>

+ **1、查看系统版本**

**内核版本与系统版本:**

Linux其实就是一个操作系统最底层的核心及其提供的核心工具。Linux也用了很多的GNU相关软件，所以Stallman认为Linux的全名应该称之为GNU/Linux。

很多的商业公司或非营利团体，就将Linux Kernel(含tools)与可运行的软件整合起来，加上自己具有创意的工具程序,这个工具程序可以让用户以光盘/DVD或者透过网络直接安装/管理Linux系统。
 这个Kernel+Softwares+Tools的可安装程序我们称之为Linux distribution（Linux发行版本）

**查看内核版本命令：**

~~~
1、cat /proc/version
[root@localhost ~]# cat /proc/version
Linux version 2.6.18-194.8.1.el5.centos.plus (mockbuild@builder17.centos.org)
(gcc version 4.1.2 20080704 (Red Hat 4.1.2-48)) #1 SMP Wed Jul 7 11:50:45 EDT 2010

2、uname -a
[root@localhost ~]# uname -a
Linux localhost.localdomain 2.6.18-194.8.1.el5.centos.plus #1 SMP Wed Jul 7 11:50:45 EDT 2010 i686 i686 i386 GNU/Linux
~~~

**查看系统版本命令：**

~~~
1、lsb_release -a，即可列出所有版本信息：
[root@localhost ~]# lsb_release -a
LSB Version: :core-3.1-ia32:core-3.1-noarch:graphics-3.1-ia32:graphics-3.1-noarch
Distributor ID: CentOS
Description: CentOS release 5.5 (Final)
Release: 5.5
Codename: Final
这个命令适用于所有的Linux发行版，包括Redhat、SuSE、Debian…等发行版。

2、cat /etc/redhat-release，这种方法只适合Redhat系的Linux：
[root@localhost ~]# cat /etc/redhat-release
CentOS release 5.5 (Final)

[root@localhost ~]# cat /etc/issue
CentOS release 5.5 (Final)
Kernel \r on an \m
~~~

* **2、软件安装**

**因为linux安装软件的方式比较多，所以没有一个通用的办法能查到某些软件是否安装了。总结起来就是这样几类：**

1、rpm包安装的，可以用rpm -qa看到，如果要查找某软件包是否安装，用 rpm -qa | grep “软件或者包的名字”。

[root@hexuweb102 ~] rpm -qa | grep ruby

2、以deb包安装的，可以用dpkg -l能看到。如果是查找指定软件包，用dpkg -l | grep “软件或者包的名字”；

[root@hexuweb102~]dpkg-l|grepruby

3、yum方法安装的，可以用yum list installed查找，如果是查找指定包，命令后加 | grep “软件名或者包名”；

[root@hexuweb102 ~] yum list installed | grep ruby

4、如果是以源码包自己编译安装的，例如.tar.gz或者tar.bz2形式的，这个只能看可执行文件是否存在了，

上面两种方法都看不到这种源码形式安装的包。如果是以root用户安装的，可执行程序通常都在/sbin:/usr/bin目录下。

说明：其中rpm yum 是Redhat系linux的软件包管理命令，dpkg是debian系列的软件包管理命令

**Linux软件安装方式：**

~~~
1.apt，rpm，yum；

2.源代码安装；

3.二进制安装。
~~~

**1. apt，rpm，yum软件安装方式：**

**①APT方式（apt是Ubuntu的软件包管理工具）**

介绍：

apt(Advanced Packaging Tool)高级包装工具，软件包管理器

例，apt-get isntall w3m

当你在执行安装操作时，首先apt-get 工具会在本地的一个数据库中搜索关于 w3m 软件的相关信息，并根据这些信息在相关的服务器上下载软件安装，这里大家可能会一个疑问：既然是在线安装软件，为啥会在本地的数据库中搜索？要解释这个问题就得提到几个名词了：

① 软件源镜像服务器

② 软件源

我们需要定期从服务器上下载一个软件包列表，使用 sudo apt-get update 命令来保持本地的软件包列表是最新的（有时你也需要手动执行这个操作，比如更换了软件源），而这个表里会有软件依赖信息的记录，对于软件依赖，我举个例子：我们安装 w3m 软件的时候，而这个软件需要 libgc1c2 这个软件包才能正常工作，这个时候 apt-get 在安装软件的时候会一并替我们安装了，以保证 w3m 能正常的工作。

安装方式：

（1）普通安装：apt-get install softname1 softname2 …;

（2）修复安装：apt-get -f install softname1 softname2... ;(-f 是用来修复损坏的依赖关系)

（3）重新安装：apt-get --reinstall install softname1 softname2...;

卸载方式：

（1）移除式卸载：apt-get remove softname1 softname2 …;（移除软件包，当包尾部有+时，意为安装）

（2）清除式卸载 ：apt-get --purge remove softname1 softname2...;(同时清除配置)

（3） 清除式卸载：apt-get purge sofname1 softname2...;(同上，也清除配置文件)

* deb（Debian）包---Ubuntu等Linux发行版的软件安装包



**②rpm方式（Red Hat Linux的软件包管理工具）**

介绍：rpm原本是Red Hat Linux发行版专门用来管理Linux各项套件的程序

语法：rpm [选项] [软件包]

安装方式：

rpm -ivh example.rpm 

安装 example.rpm 包并在安装过程中显示正在安装的文件信息及安装进度；

查询方式：

RPM 查询操作

rpm -q …

卸载方式：

rpm -e 需要卸载的安装包

在卸载之前，通常需要使用rpm -q …命令查出需要卸载的安装包名称。

升级方式：

rpm -U 需要升级的包

**③yum方式（Yellow dog Updater, Modified）**

yum命令是在Fedora和RedHat以及SUSE中基于rpm的软件包管理器，它可以使系统管理人员交互和自动化地更细与管理RPM软件包，能够从指定的服务器自动下载RPM包并且安装，可以自动处理依赖性关系，并且一次安装所有依赖的软体包，无须繁琐地一次次下载、安装。

**yum和rpm的区别：**

①   rpm软件包形式的管理虽然方便，但是需要手工解决软件包的依赖关系。很多时候安装一个软件安装一个软件需要安装1个或者多个其他软件，手动解决时，很复杂，yum解决这些问题。Yum是rpm的前端程序，主要目的是设计用来自动解决rpm的依赖关系，其特点：

1） 自动解决依赖关系；2）可以对rpm进行分组，基于组进行安装操作；3）引入仓库概念，支持多个仓库；4）配置简单

②   yum仓库用来存放所有的现有的.rpm包，当使用yum安装一个rpm包时，需要依赖关系，会自动在仓库中查找依赖软件并安装。仓库可以是本地的，也可以是HTTP、FTP、nfs形式使用的集中地、统一的网络仓库。

③   仓库的配置文件/etc/yum.repos.d目录下

安装：

yum install package1 #安装指定的安装包package1

卸载：

yum remove package1

**2. 源代码安装方式：**

源码安装（.tar、tar.gz、tar.bz2、tar.Z）

首先解压缩源码压缩包然后通过tar命令来完成

a．解xx.tar.gz：tar zxf xx.tar.gz

b．解xx.tar.Z：tar zxf xx.tar.Z

c．解xx.tgz：tar zxf xx.tgz

d．解xx.bz2：bunzip2 xx.bz2

e．解xx.tar：tar xf xx.tar

然后进入到解压出的目录中，建议先读一下README之类的说明文件，因为此时不同源代码包或者预编译包可能存在差异，然后建议使用ls -F --color或者ls -F命令（实际上我的只需要 l 命令即可）查看一下可执行文件，可执行文件会以*号的尾部标志。

一般依次执行：

./configure （检查编译环境）

make （对源代码进行编译）

sudo make install （将生成的可执行文件安装到当前计算机中）

make clean (选择执行，主要是用来清除一些临时文件)

即可完成安装。

**3. 二进制安装方式：**

二进制包的安装比较简单，我们需要做的只是将从网络上下载的二进制包解压后放到合适的目录，然后将包含可执行的主程序文件的目录添加进PATH环境变量即可，如果你不知道该放到什么位置，请重新复习第四节关于 Linux 目录结构的内容。

# 3. 总结 <span id="3">  





# 4. 参考列表 <span id="4">  

[整理|Linux下查看系统版本号的方法](https://www.jianshu.com/p/80711145e345)

[Linux下安装软件的3种方式](https://www.jianshu.com/p/0490e5208442)

[linux_查看已安装程序](https://blog.csdn.net/mikyz/article/details/69397698?utm_medium=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase)