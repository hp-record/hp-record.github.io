---
title: 网络训练（Network Train）知识的相关总结
layout: post
categories: CNN 
tags: Network-Train
excerpt: 目前对网络训练的了解，随时补充
---

---------

# 目录 <span id="home">

* **[1. 引言](#1)**
* **[2. 主要内容](#2)**
  * **[2.1 ](#2.1)**
  * **[2.2 ](#2.2)**
  * **[2.3 ](#2.3)**
* **[3. 总结](#3)**
* **[4. 参考列表](#4)**

---------

# 1. 引言 <span id="1">  

这篇文章旨在整理深度学习网络训练相关知识的个人理解，通过学习网络训练的好文以及自己的网络训练实践理解，好好整理，以供个人知识梳理及分享给有需要的人们(❤ ω ❤)。

# 2. 主要内容<span id="2">  

**2.1 卷积网络的训练过程**

卷积神经网络的训练过程分为两个阶段。第一个阶段是数据由低层次向高层次传播的阶段，即前向传播阶段。另外一个阶段是，当前向传播得出的结果与预期不相符时，将误差从高层次向底层次进行传播训练的阶段，即反向传播阶段。训练过程如下图所示。

![Network-Train.png](https://i.loli.net/2020/06/15/tuOXownUsbD1I6j.png)

训练过程为：

1、网络进行权值的初始化；

2、输入数据经过卷积层、下采样层、全连接层的向前传播得到输出值；

3、求出网络的输出值与目标值之间的误差；

4、当误差大于我们的期望值时，将误差传回网络中，依次求得全连接层，下采样层，卷积层的误差。各层的误差可以理解为对于网络的总误差，网络应承担多少；当误差等于或小于我们的期望值时，结束训练。

5、根据求得误差进行权值更新。然后在进入到第二步。



**2.2 基础概念**

| 名词      | 定义                                                         |
| --------- | ------------------------------------------------------------ |
| Epoch     | 使用训练集的全部数据对模型进行一次完整训练，被称为 “一代训练” |
| Batch     | 使用训练集中的一部分样本对模型权重进行一次反向传播的参数更新，这一小部分样本被称为 “一批数据” |
| Iteration | 使用一个Batch数据对模型进行一次参数更新的过程，被称为 “一次训练” |

epoch：训练时，所有训练数据集都训练过一次。

batch_size：在训练集中选择一组样本用来更新权值。1个batch包含的样本的数目，通常设为2的n次幂，常用的包括64,128,256。 网络较小时选用256，较大时选用64。

iteration：训练时，1个batch训练图像通过网络训练一次（一次前向传播+一次后向传播），每迭代一次权重更新一次；测试时，1个batch测试图像通过网络一次（一次前向传播）。所谓iterations就是完成一次epoch所需的batch个数。

三者之间的关系：iterations = epochs×(images / batch_size)，所以1个epoch包含的iteration次数=样本数量/batch_size。



换算公示： $Numbers-of-Batches = \frac {Training-Set-Size}{Batch-Size}$

 实际上，梯度下降的几种方式的根本区别就在于上面公式中的 Batch Size不同。

| 梯度下降方式 | Training Set Size | Batch Size | Numbers of Batches |
| ------------ | ----------------- | ---------- | ------------------ |
| BGD          | N                 | N          | 1                  |
| SGD          | N                 | 1          | N                  |
| Mini-Batch   | N                 | B          | N/B + 1            |

*注：上表中 Mini-Batch 的 Batch 个数为 N / B + 1 是针对未整除的情况。整除则是 N / B。

举例：

CIFAR10 数据集有 50000 张训练图片，10000 张测试图片。现在选择 Batch Size = 256 对模型进行训练。

- **每个 Epoch 要训练的图片数量：**![50000](https://www.zhihu.com/equation?tex=50000)
- **训练集具有的 Batch 个数：**![50000 / 256 = 195 + 1 = 196](https://www.zhihu.com/equation?tex=50000+%2F+256+%3D+195+%2B+1+%3D+196)
- **每个 Epoch 需要完成的 Batch 个数：**![196](https://www.zhihu.com/equation?tex=196)
- **每个 Epoch 具有的 Iteration 个数：**![196](https://www.zhihu.com/equation?tex=196)
- **每个 Epoch 中发生模型权重更新的次数：**![196](https://www.zhihu.com/equation?tex=196)
- **训练** ![10](https://www.zhihu.com/equation?tex=10) **代后，模型权重更新的次数：**![196 * 10 = 1960](https://www.zhihu.com/equation?tex=196+%2A+10+%3D+1960)
- **不同代的训练，其实用的是同一个训练集的数据。第** ![1](https://www.zhihu.com/equation?tex=1) **代和第** ![10](https://www.zhihu.com/equation?tex=10) **代虽然用的都是训练集的五万张图片，但是对模型的权重更新值却是完全不同的。因为不同代的模型处于代价函数空间上的不同位置，模型的训练代越靠后，越接近谷底，其代价越小。**

# 3. 总结 <span id="3">  







# 4. 参考列表 <span id="4">  

[卷积神经网络的训练过程](https://zhuanlan.zhihu.com/p/36627246)

[训练神经网络中最基本的三个概念：Epoch, Batch, Iteration](https://zhuanlan.zhihu.com/p/29409502)







