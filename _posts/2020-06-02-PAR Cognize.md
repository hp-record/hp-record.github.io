---
title: 行人属性识别(PAR)-综述
layout: post
categories: PAR
tags: 综述
excerpt: 目前对PAR的了解，随时补充
---

---------

# 目录 <span id="home">

* **[1. 引言](#1)**
* **[2. PAR](#2)**
  * **[2.1 历史发展](#2.1)**
  * **[2.2 现状评述](#2.2)**
    * **[2.2.1 PAR-Benchmarks](#2.2.1)**
    * **[2.2.2 PAR-Regular Pipeline](#2.2.2) **
    * **[2.2.3  PAR-Algorithms](#2.2.3)**
      * **[2.2.3.1  经典的深度神经网络](#2.2.3.1)**
      * **[2.2.3.2  一些PAR算法](#2.2.3)**
  * **[2.3 发展前景预测](#2.3)**
* **[3. 总结](#3)**
* **[4. 参考列表](#4)**

---------

#  1. 引言 <span id="1">   

目前，在对行人属性识别进行了解的时候，对它的了解都很片面，没有一个全面宏观的前世今生的认识，于是对行人属性识别相关查到的认识做个总结整理，易便理解。

首先，**什么是行人属性识别（Pedestrian Attribute Recognition, PAR）？**简单来说，行人属性识别就是给定行人图像时挖掘行人目标的属性，即对图像里的行人目标进行检测与提取，识别行人的特征（比如性别、年龄；是否留胡须、戴口罩、戴眼镜；上下衣颜色、戴帽子；是否有携带物等特征） 。

![attributed parse graph.jpg](https://i.loli.net/2020/06/02/FAtVKjw5WGuXa3h.jpg)

其次，PAR主要应用于**自然场景**和**监控场景**中，用于行人ReID、人员检测等技术中。精准的人体属性信息，能够帮助开展各类基于人体照片的分析工作，已被广泛运用于刑事侦查、广告精准投放与商业零售市场研究分析等领域。但是尽管已经提出了许多关于PAR的工作，但是由于各种Challenging(挑战性)因素，例如**视点变化**，**低照度**，**低分辨率**等，行人属性识别仍然是一个尚未解决的问题。

**回顾PAR方法的演进**，传统的行人属性识别方法通常着重于从手工选取特征，强大的分类器或属性关系的角度来开发鲁棒的特征表示。一些好的特征应用包括HOG，SIFT ，SVM 或CRF模型。但是，大型基准评估的报告表明，这些传统算法的性能离实际应用的要求还差的很远。在过去的几年中，深度学习取得了骄人的成绩，这是因为它们在使用多层非线性变换进行自动特征提取方面取得了成功，尤其是在计算机视觉，语音识别和自然语言处理方面。基于这些突破，提出了一些基于深度学习的属性识别算法，大都是基于深度学习的**多标签(或多任务)学习**的方法。

# 2. PAR <span id="2">

> 这部分，从以下三个部分总结整理PAR：

> ​	（1）历史发展：按时间顺序，简述该主题的来龙去脉，发展概况及各阶段的研究水平。

> ​	（2）现状评述：重点是论述当前国内外的研究现状，着重评述哪些问题已经解决，哪些问题还没有解决，提出可能的解决途径；存在的争论焦点，比较各种观点的异同并作出理论解释，亮明作者的观点；详细介绍有创造性和发展前途的理论和假说，并引出论据，指出可能的发展趋势。

> ​	（3）发展前景预测：通过纵横对比，肯定该主题的研究水平，指出存在的问题，提出可能的发展趋势，指明研究方向，提示研究的捷径。

## 2.1 历史发展 <span id="2.1">

> 以自己学习了解到的程度整理总结下了相关的认知，如有错误，敬请斧正.		

行人属性识别在之前采用的是传统的图像识别方法，之后在11年12年之后深度学习的滚滚洪流奔袭而来，近几年提出了一些基于深度学习的属性识别算法。

·首先，先简要整理介绍下**传统的方法**模式。

传统的方法使用滑动窗口的框架，把一张图分解为许多个不同位置不同尺度的子窗口，针对每一个窗口使用分类器判断是否包含目标物体（即**特征提取+分类识别**）。

传统方法针对不同类别的物体，一般都要设计不同的特征和分类算法，比如人脸检测是经典算法是Harr特征+Adaboosting分类器；行人检测的经典算法HOG(histogram of gradients) + SVM；一般性物体的检测的话是HOG的特征加上DPM(deformable part model)的算法。

故，传统的行人属性识别方法通常是从设计选取手工特征、强大的分类器或属性关系的角度开发鲁棒性较好的特征表示，一些经典的特征包括HOG、SIFT、SVM或CRF模型等。但是传统方法中特征提取主要依赖人工设计的提取器，需要有专业知识及复杂的调参过程，同时每个方法都是针对具体应用，泛化能力及鲁棒性较差。它大致的缺点总结如下：

（1）大量冗余的proposal生成，导致**学习效率低下**，容易在分类出现大量的假正样本。

（2）特征描述子都是基于低级特征进行手工设计的，**难以捕捉高级语义特征**（如行人属性）和复杂内容。

（3）检测的每个步骤是独立的，**缺乏一种全局的优化方案**进行控制。

所以，传统的行人属性识别方法离实际应用需要的性能差的很远。

·再看看近几年被认为主流的**深度学习方法**。

深度学习的方法，引入了**端到端学习**的概念，即向机器提供的图像数据集中的每张图像均已标注目标类别，然后输出新图像中的目标类别。

因此深度学习模型基于给定的数据【训练】得到【特征】，其中神经网络用来发现图像类别中的底层模式，并自动提取对于目标类别（或者属性）最具描述性和最具显著的特征。

在过去的几年中，深度学习取得了骄人的成绩，人们普遍认为 DNN(深度神经网络) 的性能大大超过传统算法，随着 CV 领域中最优秀的方法纷纷使用深度学习，CV 工程师的工作流程出现巨大改变，手动提取特征所需的知识和专业技能被使用深度学习架构进行迭代所需的知识和专业技能取代（见下图-(a)传统CV 工作流程   (b)深度学习工作流程）。

<img src="https://i.loli.net/2020/05/18/wpYoemQxSt5EkrF.jpg" alt="传统vs深度学习（流程）.jpeg" style="zoom:67%;" />

·目前，行人属性识别仍然是一个具有挑战性的问题。

行人属性识别，在给定一个行人目标，旨在根据预定义的属性列表A={a1, a2, ..., aL}来预测一组描述该人特征的属性特征。如下图所示。

![PAR.jpg](https://i.loli.net/2020/05/18/bxIByLnTOtWlqJD.jpg)

但是，由于属性类别的类内差异很大（外观多样性和外观模糊性），该任务仍然具有挑战性。

列出了以下可能明显影响最终属性识别性能的**挑战性因素**，如下所示：

（1）多视点（multi-view）：相机从不同角度拍摄的图像导致许多计算机视觉任务出现视点问题，由于人的身体刚性，这进一步使人的属性识别更加复杂。

（2）遮挡（Occlusion）：人体的一部分被其他人或者物体遮挡，会增加人的属性识别的难度。因为被遮挡的部分引入的像素值可能会使模型混乱，并导致错误的预测。

（3）数据分布不平衡（Unbalanced Data Distribution）：每个人都有不同的属性，因此，属性的数量是可变的，这导致数据分配不平衡。众所周知，现有的机器学习算法在这些数据集上表现不佳。

（4）低分辨率（Low Resolution）：在实际情况下，由于高品质的相机成本高昂，因此获取的图像的分辨率一般较低。因此，在这种条件进行人的属性识别，本身就有难度。

（5）光照（Illumination）：这些图像来自于在24小时内的任何时间拍摄。因此，光照条件在不同时间可能不同。阴影可能还会出现在人像中，而且夜间拍摄的图像可能完全无效。

（6）模糊（Blur）：当人在移动时，相机拍摄到的图像可能会出现模糊。

因此在这些情况下如何正确识别人的属性是一项非常具有挑战性的任务。

## 2.2 现状评述 <span id="2.2">

> 这一块，从以下3个方面来总结陈述PAR的研究现状。
>
> (1)PAR-Benchmarks
>
> (Benchmark：that is used as a standard by which other things can be judged or measured.)
>
> (2)PAR-Regular Pipeline
>
>  (3)PAR-Algorithms

### 2.2.1 PAR-Benchmarks <span id="2.2.1">

**A.数据集(Dataset)**

与计算机视觉中的其他任务不同，对于行人属性识别，数据集的注释包含许多**不同级别的标签**。

例如，发型和颜色，帽子，眼镜等被视为特定的低级属性，并且对应于图像的不同区域，而某些属性是抽象概念，例如性别，方向和年龄，它们并不对应对于某些区域，我们将这些属性视为高级属性。此外，人类属性识别通常受到环境或上下文因素的严重影响，例如视点，遮挡物和身体部位。为了便于研究，一些数据集提供者的视角，部位边界框，遮挡物等属性。

回顾近年来的PAR相关工作，总结整理一些用于研究行人属性识别的数据集，包括PETA ，RAP ，RAP-2.0 ，PA-100K ，WIDER，Market-1501，DukeMTMC ，Clothing Attributes Dataset，PARSE-27K，APiS，HAT，Berkeley-Attributes of People dataset和CRP dataset。[·数据集来源](https://www.cnblogs.com/geoffreyone/p/10336919.html)

**B.评估标准(Evaluation Criteria)**

·一般是由ROC曲线和AUC来评估每个属性分类的性能。

**ROC曲线**

ROC曲线---接收者操作特征曲线（receiver operating characteristic，简称ROC），是一种显示分类模型在所有分类阈值下的效果的图表。该曲线绘制了以下两个参数：

- 真正例率
- 假正例率

真正例率（TPR）是召回率的同义词，定义：$TPR=\frac{TP}{TP+FN}$

假正例率（FPR）是误报率的同义词，定义：$FPR=\frac{FP}{FP+TN}$

ROC 曲线用于绘制采用不同分类阈值时的 TPR 与 FPR。

降低分类阈值会导致将更多样本归为正类别，从而增加假正例和真正例的个数。

下图显示了一个典型的 ROC 曲线。

![ROCCurve.jpg](https://i.loli.net/2020/05/19/tHvM5DkQ3RJBSoK.jpg)

**AUC**

AUC---ROC曲线下面积（Area under the ROC Curve，简称AUC），也就是说，曲线下面积测量的是从 (0,0) 到 (1,1) 之间整个 ROC 曲线以下的整个二维面积。

曲线下面积对所有可能的分类阈值的效果进行综合衡量。曲线下面积的一种解读方式是看作模型将某个随机正类别样本排列在某个随机负类别样本之上的概率。

<img src="https://i.loli.net/2020/05/19/Ihy8mTwAHocFOln.jpg" alt="AUC.jpg" style="zoom:50%;" />

曲线下面积的取值范围为 0-1。预测结果 100% 错误的模型的曲线下面积为 0.0；而预测结果 100% 正确的模型的曲线下面积为 1.0。

·一般对于属性识别算法采用平均精度（mA）来评估。

对于每个属性，mA分别计算正样本和负样本的分类精度，然后获取其平均值作为该属性的识别结果。最后，通过对所有属性取平均值来获得识别率。通过以下公式计算评估标准：
$$
mA=\frac{1}{2N}\sum_{i=1}^{L}{(\frac{TP_i}{P_i}+\frac{TN_i}{N_i})}
$$
其中L是属性的数量。TPi和TNi分别是正确预测的正例和负例的数目，Pi和Ni分别是正例和负例的数目。

·以上的评估标准（被称为基于标签的的评估标准，label-based evaluation criterions）独立的对待每个属性，而忽略了在多属性识别问题中自然存在的属性间相关性。

对于行人属性识别问题采用基于示例的评估标准(example-based evaluation criterions)，包括以下四个指标：准确率，精确率，召回率和F1值。
$$
Acc_(exam)=\frac{1}{N}\sum_{i=1}^{N}{|\frac{Y_i \bigcap f(x_i)}{Y_i \bigcup f(x_i)}|}
$$

$$
Prec_(exam)=\frac{1}{N}\sum_{i=1}^{N}{\frac{|Y_i \bigcap f(x_i)|}{|Y_i| \bigcup f(x_i)|}}
$$

$$
Rec_(exam)=\frac{1}{N}\sum_{i=1}^{N}{\frac{|Y_i \bigcap f(x_i)|}{|Y_i|}}
$$

$$
F1=\frac{2*Prec_(exam)*Rec_(exam)}{Prec_(exam)+Rec_(exam)}
$$

其中，N是样本数，Yi是第i个正确的正标签的样本，f(x)返回的是对于第i个样本预测的正标签。|·|表示集的数量。

### 2.2.2 PAR-Regular Pipeline<span id="2.2.2">

行人属性识别，并不是直观的采用学习每个属性然后再输出的机制，因为这样会使得识别效率低下而且也会与行人属性这个整体概念不符。因此，目前研究人员倾向于在一个模型中预测所有属性，并将每个属性预测视为一项任务。也就是多任务学习框架。另一方面，行人属性识别模型将给定的行人图像作为输入，然后输出相应的属性，从这个角度讲，PAR也属于多标签学习框架。

·多任务学习与其他学习算法之间的关系

多任务学习（Multitask learning）是迁移学习算法的一种。定义一个一个源领域source domain和一个目标领域（target domain），在source domain学习，并把学习到的知识迁移到target domain，提升target domain的学习效果（performance）。

多标签学习（Multilabel learning）是多任务学习中的一种，建模多个label之间的相关性，同时对多个label进行建模，多个类别之间共享相同的数据/特征。

多类别学习（Multiclass learning）是多标签学习任务中的一种，对多个相互独立的类别（classes）进行建模。这几个学习之间的关系如下图所示：

![multi-task.jpg](https://i.loli.net/2020/05/21/uIciws2bD8XSKox.jpg)

目前来讲，行人属性识别的常规流程框架是采用的多任务学习（或者）多标签学习框架。以下整理介绍下这两个流程框架（pipeline）。

**·多任务学习（multi-task learning）**

·一般在机器学习（或者深度学习）领域解决一项特别任务，传统的解决方案是设计评估标准，提取相关特征描述并构造单个或集成模型，然后使用特征描述符来优化模型参数，并根据评估标准获得最佳结果，从而改善整体性能。

该管道框架（pipeline）可以在单个任务上获得令人满意的结果，但是，它对于评估标准来说，忽略了其他任务可能带来的进一步改进。

在现实世界中，许多事物是相关的。学习一项任务可能依赖或约束其他任务。即使分解一个任务，但子任务在某种程度上仍具有相关性。独立处理单个任务很容易忽略这种相关性，因此，最终性能的提高可能会遇到瓶颈。具体而言，行人属性彼此相关，例如性别和衣服样式。另一方面，监督学习需要大量难以收集的带注释的训练数据。因此，最流行的方法是联合学习多任务以挖掘共享特征表示。这个管道框架已被广泛应用于自然语言处理，计算机视觉等多个领域。随着深度学习的发展，通过集成多任务学习和深度神经网络，提出了许多有效的算法。

那什么是多任务学习（multi-task learning）？简单讲，基于共享表示（shared representation），把多个相关的任务放在一起学习的机器学习方法框架。

共享表示（shared representation）的目的是为了提高泛化（improving generalization），多个任务在浅层共享参数。MTL中共享表示由两种方式：硬参数共享（也叫基于参数的共享，parameter based）、软参数共享（也叫基于约束的共享，regularization based）。

·有学者研究，为什么多个相关任务放在一起学习，可以提高学习的效果？其原因归纳为以下几点：

（1）多人相关任务放在一起学习，有相关的部分，但也有不相关的部分。当学习一个任务（Main task）时，与该任务不相关的部分，在学习过程中相当于是噪声，因此，引入噪声可以提高学习的泛化（generalization）效果。

（2）单任务学习时，梯度的反向传播倾向于陷入局部极小值。多任务学习中不同任务的局部极小值处于不同的位置，通过相互作用，可以帮助隐含层逃离局部极小值。

（3）添加的任务可以改变权值更新的动态特性，可能使网络更适合多任务学习。比如，多任务并行学习，提升了浅层共享层（shared representation）的学习速率，可能，较大的学习速率提升了学习效果。

（4）多个任务在浅层共享表示，可能削弱了网络的能力，降低网络过拟合，提升了泛化效果。

还有很多潜在的解释，多任务学习有效，是因为它是建立在多个相关的，具有共享表示（shared representation）的任务基础之上的，因此，需要定义一下，什么样的任务之间是相关的。

相关（related）的具体定义很难，但我们可以知道的是，在多任务学习中，related tasks可以提升main task的学习效果，基于这点得到相关的定义：

Related（Main Task，Related tasks，LearningAlg）= 1    （1）

LearningAlg（Main Task||Related tasks）> LearningAlg（Main Task） （2）

LearningAlg表示多任务学习采用的算法，第一个公式表示，把Related tasks与main tasks放在一起学习，效果更好；第二个公式表示，基于related tasks，采用LearningAlg算法的多任务学习Main task，要比单学习main task的条件概率概率更大。

特别注意，相同的学习任务，基于不同学习算法，得到相关的结果不一样：

Related（Main Task，Related tasks，LearningAlg1）不等于 Related（Main Task，Related tasks，LearningAlg2）

·多任务学习并行学习时，有5个相关因素可以帮助提升多任务学习的效果。

（1）隐式数据扩充（data amplification）。相关任务在学习过程中产生的额外有用的信息可以有效数据/样本（data）的大小/效果。主要有三种数据放大类型：统计数据放大（statistical data amplification）、采样数据放大（sampling data amplification），块数据放大（blocking data amplification）。

（2）Eavesdropping（窃听）。

（3）属性选择（attribute selection）

（4）表示偏移（representation bias）

（5）预防过拟合（overfitting prevention）

所有这些关系（relationships）都可以帮助提升学习效果（improve learning performance）。

·多任务学习的两种共享表示：

硬参数共享通常将浅层作为共享层，以学习多个任务的通用特征表示，而将高级层视为特定于任务的层，以学习更多区分模式。该模式是深度学习社区中最受欢迎的框架。如下图所示：

<img src="https://i.loli.net/2020/05/21/dzFa3X4noCbJhyB.png" alt="多任务学习-硬参数共享.png" style="zoom: 50%;" />

<center>图-多任务学习-硬参数共享</center>

对于软参数共享多任务学习（如下图所示），他们独立地训练每个任务，但是通过引入的正则化约束（例如L2距离）使不同任务之间的参数相似，[L2距离] 并追踪正则化。

简单来讲：每个任务有自己的参数，最后**通过对不同任务的参数之间的差异加约束**，表达相似性。比如可以使用L2, trace norm等。

<img src="https://i.loli.net/2020/05/21/7NiSdhw4Pe8FmqK.png" alt="多任务学习-软参数共享.png" style="zoom:50%;" />

<center>图-多任务学习-软参数共享</center>

·多标签学习（multi-label learning）

传统监督学习主要是单标签学习，而现实生活中目标样本往往比较复杂，具有多个语义，含有多个标签。具体而言，对于行人属性目标来说，它就具有多个语义，含有多个标签。

1、学习任务

X=R^d表示d维的输入空间，Y={y1,y2,.....,yq}表示带有q个可能标签的标签空间。训练集$D = {(x^i, y^i)| 1 \leq i \leq m} ，m表示训练集的大小，上标表示样本序数，有时候会省略。x^i \in X，是一个d维的向量。y^i \subseteq Y，是Y的一个标签子集。


任务就是要学习一个多标签分类器h(\cdot )，预测h(x) \subseteq Y作为x的正确标签集。常见的做法是学习一个衡量x和y相关性的函数，希望f(x, y_{j1}) > f(x, y_{j2})，其中y_{j1} \in y, y_{j2} \notin y。h(x)可以由f(x)衍生得到，h(x) = {y_j | f(x,y_j) > t(x), y_j \in Y}。

t(x)扮演阈值函数的角色，把标签空间对分成相关的标签集和不相关的标签集。阈值函数可以由训练集产生，可以设为常数。当f(x, y_j)$返回的是一个概率值时，阈值函数可设为常数0.5。

2、三种策略

多标签学习的主要难点在于输出空间的爆炸增长，比如20个标签，输出空间就有2^20，为了应对指数复杂度的标签空间，需要挖掘标签之间的相关性。比方说，一个图像被标注的标签有热带雨林和足球，那么它具有巴西标签的可能性就很高。一个文档被标注为娱乐标签，它就不太可能和政治相关。有效的挖掘标签之间的相关性，是多标签学习成功的关键。根据对相关性挖掘的强弱，可以把多标签算法分为三类。

- 一阶策略：忽略和其它标签的相关性，比如把多标签分解成多个独立的二分类问题（简单高效）。
- 二阶策略：考虑标签之间的成对关联，比如为相关标签和不相关标签排序。
- 高阶策略：考虑多个标签之间的关联，比如对每个标签考虑所有其它标签的影响（效果最优）。

评价指标：（可分为两类）

·基于样本的评价指标（先对单个样本评估表现，然后对多个样本取平均）

·基于标签的评价指标（先考虑单个标签在所有样本上的表现，然后对多个标签平均）

![multi-label evaluation metrics.png](https://i.loli.net/2020/05/21/1Z2CYioQGyKBWVg.png)

学习算法： （可分为两类）

·问题转换的方法：把多标签问题转为其他学习场景，比如

·算法改编的方法：通过改编流行的学习算法直接处理多标签数据，比如改编懒学习，决策树，核技巧。

![multi-label alg.png](https://i.loli.net/2020/05/21/6yHgmcOe2TilhAd.png)

### 2.2.3 PAR-Algorithms <span id="2.2.3">

在介绍算法之前，先回顾下常用(或者叫经典)的深度神经网络.

#### 2.2.3.1  经典的深度神经网络<span id="2.2.3.1">

1.LeNet

·介绍：

LeNet，是最早的卷积神经网络之一，并且推动了深度学习领域的发展。自从 1988 年开始，在许多次成功的迭代后，这项由 Yann LeCun（深度学习三巨头之一，同时也是卷积神经网络 (CNN，Convolutional Neural Networks)之父。） 完成的开拓性成果被命名为 LeNet5。（1998年，在发表的论文中回顾了应用于手写字符识别的各种方法，并用标准手写数字识别基准任务对这些模型进行了比较，结果显示卷积神经网络的表现超过了其他所有模型。LeCun, Y.; Bottou, L.; Bengio, Y. & Haffner, P. (1998). Gradient-based learning applied to document recognition.Proceedings of the IEEE. 86(11): 2278 - 2324.）

LeNet主要用来进行**手写和机器打印的字符的识别与分类**。LeNet的实现确立了CNN的结构，现在神经网络中的许多内容在LeNet的网络结构中都能看到，例如**卷积层**，**Pooling层**，**ReLU层**。虽然LeNet早在20世纪90年代就已经提出了，但由于当时缺乏大规模的训练数据，计算机硬件的性能也较低，因此LeNet神经网络在处理复杂问题时效果并不理想。现在在研究中已经很少将LeNet使用在实际应用上，对卷积神经网络的设计往往在某个或多个方向上进行优化，如包含更少的参数（以减轻计算代价）、更快的训练速度、更少的训练数据要求等。

·网络结构：

LeNet-5包含七层，不包括输入，每一层都包含可训练参数（权重），当时使用的输入数据是32*32像素的图像。

网络结构图如下：（[详细介绍](https://www.jiqizhixin.com/graph/technologies/6c9baf12-1a32-4c53-8217-8c9f69bd011b)）

![LeNet5.jpeg](https://i.loli.net/2020/05/24/ysivjenwHcKuFdp.jpg)

2.AlexNet

·介绍：

AlexNet（以第一作者alex命名）是2012年ImageNet竞赛冠军获得者Hinton和他的学生Alex Krizhevsky设计的，在 ImageNet LSVRC-2010 数据集上对1000个类别的图像进行分类取得了当时最好的效果；同时在 ILSVRC-2012 数据集上取得了当时第一的成绩,是深度学习历史上的一个里程碑。在那年之后，更多的更深的神经网络被提出，比如优秀的vgg,GoogLeNet。

AlexNet中包含了几个比较新的技术点，也首次在CNN中成功应用了ReLU、Dropout和LRN（局部响应归一化）等Trick。同时AlexNet也使用了GPU进行运算加速。AlexNet将LeNet的思想发扬光大，把CNN的基本原理应用到了很深很宽的网络中。

说实话，这个model的意义比后面那些model都大很多，首先它证明了CNN在复杂模型下的有效性，然后GPU实现使得训练在可接受的时间范围内得到结果，确实让CNN和GPU都大火了一把，顺便推动了有监督DL的发展。

·网络结构：

AlexNet网络一共有八层，包括5个卷积层与三个全连接层。([详情1](https://zhuanlan.zhihu.com/p/59524479)、[详情2](https://zhuanlan.zhihu.com/p/42914388))

![AlexNet-0.jpg](https://i.loli.net/2020/05/24/6i9Jku4pamhUsRA.jpg)

<center>AlexNet-论文原文中的图</center>

<img src="https://i.loli.net/2020/05/24/RaWZs8ApKPihOTv.jpg" alt="AlexNet-1.jpg" style="zoom:80%;" />

<center>AlexNet-细化的结构图</center>

3.VGG

·介绍：

[VGG](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1409.1556)是Oxford的**V**isual **G**eometry **G**roup的组提出的（应该能看出VGG名字的由来了）。该网络是在ILSVRC 2014上的相关工作，主要工作是**证明了增加网络的深度能够在一定程度上影响网络最终的性能**。

VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

·网络结构：

[一文读懂VGG网络](https://zhuanlan.zhihu.com/p/41423739)

![VGG.jpg](https://i.loli.net/2020/05/24/YhFsU2TQek1WJMV.jpg)

\- VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示

\- VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示

4.GoogLeNet

·介绍：

2014年，GoogLeNet和VGG是当年ImageNet挑战赛(ILSVRC14)的双雄，GoogLeNet获得了第一名、VGG获得了第二名，这两类模型结构的共同特点是层次更深了。VGG继承了LeNet以及AlexNet的一些框架结构（详见 [大话CNN经典模型：VGGNet](https://my.oschina.net/u/876354/blog/1634322)），而GoogLeNet则做了更加大胆的网络结构尝试，虽然深度只有22层，但大小却比AlexNet和VGG小很多，GoogleNet参数为500万个，AlexNet参数个数是GoogleNet的12倍，VGGNet参数又是AlexNet的3倍，因此在内存或计算资源有限时，GoogleNet是比较好的选择；从模型结果来看，GoogLeNet的性能却更加优越。

GoogLeNet(也有称inception)是2014年Christian Szegedy提出的一种全新的深度学习结构，在这之前的AlexNet、VGG等结构都是通过增大网络的深度（层数）来获得更好的训练效果，但层数的增加会带来很多负作用，比如overfit、梯度消失、梯度爆炸等。inception的提出则从另一种角度来提升训练结果：能更高效的利用计算资源，在相同的计算量下能提取到更多的特征，从而提升训练结果。

·网络结构：

[大话CNN经典模型：GoogLeNet（从Inception v1到v4的演进）](https://my.oschina.net/u/876354/blog/1637819)

Inception模块结构：

![inception.jpg](https://i.loli.net/2020/05/24/B67eOibHwWadxFG.jpg)

5.Residual Network

·介绍：

Kaiming He 的《Deep Residual Learning for Image Recognition》获得了CVPR最佳论文。2015年，微软亚洲研究院的何凯明等人使用残差网络ResNet[4]参加了当年的ILSVRC，在图像分类、目标检测等任务中的表现大幅超越前一年的比赛的性能水准，并最终取得冠军。

残差网络的明显特征是有着相当深的深度，从32层到152层，深度远远超过了之前提出的深度网络结构，后又针对小数据设计了1001层的网络结构。残差网络ResNet的深度惊人，极其深的深度使该网络拥有极强大的表达能力。

残差网络首先以其超深度架构（超过1k层）而广为人知，而相比之下，先前的网络则相对“浅”。该网络的主要贡献是引入了残差块，如下图所示。该机制可以通过引入身份跳过连接并将其输入复制到下一层来解决训练真正的深度体系结构的问题。这种方法可以在很大程度上解决消失梯度问题。

![inception-1.jpg](https://i.loli.net/2020/05/24/mynxb3OKHizWBuP.jpg)

·网络结构：

[大话深度残差网络（DRN）ResNet网络原理](https://my.oschina.net/u/876354/blog/1622896)：

![ResNet.png](https://i.loli.net/2020/05/24/NxfdKp78uHFo9yi.png)

6.Dense Network

·介绍：

在计算机视觉领域，卷积神经网络（CNN）已经成为最主流的方法，比如最近的GoogLenet，VGG-19，Incepetion等模型。CNN史上的一个里程碑事件是ResNet模型的出现，ResNet可以训练出更深的CNN模型，从而实现更高的准确度。ResNet模型的核心是通过建立前面层与后面层之间的“短路连接”（shortcuts，skip connection），这有助于训练过程中梯度的反向传播，从而能训练出更深的CNN网络。

而现在要介绍的是DenseNet模型，它的基本思路与ResNet一致，但是它建立的是前面所有层与后面层的密集连接（dense connection），它的名称也是由此而来。DenseNet的另一大特色是通过特征在channel上的连接来实现特征重用（feature reuse）。这些特点让DenseNet在参数和计算成本更少的情形下实现比ResNet更优的性能，DenseNet也因此斩获CVPR 2017的最佳论文奖。

·网络结构：

[DenseNet：比ResNet更优的CNN模型](https://zhuanlan.zhihu.com/p/37189203) 下图是一个dense block示意图：



![dense-block.png](https://i.loli.net/2020/05/24/uABgPaSC4GvN1R6.png)

小结一下：

| 年份 | 事件                                                         | 相关论文/Reference                                           |
| ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1989 | Yann LeCun等人首次发表CNN网络                                | LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. *Proceedings of the IEEE*, *86*(11), 2278-2324. |
| 2012 | Alex和Hinton提出AlexNet，并在ILSVRC2012比赛中获得很好的成绩，AlexNet为卷积神经网络的开山之作 | Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). Imagenet classification with deep convolutional neural networks. In *Advances in neural information processing systems* (pp. 1097-1105). |
| 2014 | VGG网络被提出，其主要贡献是增加了网络深度                    | Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*. |
| 2014 | GoogLeNet巧妙运用inception模块来提升训练结果，并获得了ILSVRC2014的冠军 | Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., ... & Rabinovich, A. (2015). Going deeper with convolutions. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 1-9). |
| 2015 | 残差网络的提出减小了深度网络带来的一系列问题。它让网络变得更深并且提升了识别准确率 | He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 770-778). |
| 2016 | DenseNet被提出，主要贡献为实现底层与高层的feature信息的共享  | Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017, July). Densely connected convolutional networks. In *CVPR* (Vol. 1, No. 2, p. 3). |

7.Capsule Network

·介绍：

胶囊网络（Capsule Networks）是深度学习三巨头之一的Geoffrey Hinton于2017年提出的一种全新的神经网络。胶囊网络基于一种新的结构——胶囊（Capsule），通过与现有的卷积神经网络（CNN）相结合，从而在一些图像分类的数据上取得了非常优越的性能。

理解胶囊网络，首先需要抛开对CNN架构的固有印象，因为Geoffrey Hinton实际上认为在CNN中应用pooling是一个很大的错误，它工作得很好的事实是一场灾难。在最大池化过程中，很多重要的信息都损失了，因为只有最活跃的神经元会被选择传递到下一层，而这也是层之间有价值的空间信息丢失的原因。（比如说：一张图包含人脸所有特征，但是位置是乱的，固有的CNN依然识别这是人脸，Hinton认为这是不对的。）

·网络结构：

[胶囊网络](jiqizhixin.com/graph/technologies/ed5dbe49-af23-4688-8a58-74577d608c60)、[理解和使用胶囊网络](https://www.jiqizhixin.com/articles/2019-01-18-14)

![CapsuleNet.jpg](https://i.loli.net/2020/05/25/APzTZIpR5vg4qD7.jpg)

8.Graph Convolutional Network

·介绍：

2018年，Chen, J. et al.提出基于控制变量的图卷积网络（GCN），有效减少感受野大小。

图（graph）是一种数据格式，它可以用于表示社交网络、通信网络、蛋白分子网络等，图中的节点表示网络中的个体，连边表示个体之间的连接关系。许多机器学习任务例如社团发现、链路预测等都需要用到图结构数据，因此图卷积神经网络的出现为这些问题的解决提供了新的思路。

·网络结构：

[图卷积网络](https://www.jiqizhixin.com/graph/technologies/eee9ade4-80fd-4821-9dce-2dce5e898b42)、[图神经网络](https://www.jiqizhixin.com/graph/technologies/c39cf57b-df95-4c9e-9a8a-0d8ea330d625)

![Graph convolution network.png](https://i.loli.net/2020/05/25/NcC7BSqEJ21RK3F.png)

9.ReNet

·介绍：

2015年， F. Visin等人提出了一个基于循环神经网络的用于目标识别的网络结构，称为ReNet。

ReNet用基于四个方向的 RNN 来替换掉 CNN中的 convolutional layer（即：卷积+Pooling 的组合）。通过在前一层的 feature 上进行四个方向的扫描，完成特征学习的过程。

·网络结构：

[ReNet笔记](https://www.cnblogs.com/wangxiaocvpr/p/8507970.html)

![ReNet.png](https://i.loli.net/2020/05/25/9Mjx7IraGoXL25A.png)

10.Recurrent Neural Network, RNN.

·介绍：

NLP里最常用、最传统的深度学习模型就是**循环神经网络 RNN**（Recurrent Neural Network）。

递归神经网络，RNN。传统的神经网络基于所有输入和输出彼此独立的假设，但是该假设在许多任务（例如句子翻译）中可能并不正确。提出了递归神经网络（RNN）来处理涉及顺序信息的任务。 RNN之所以称为递归，是因为它们对序列的每个元素执行相同的任务，其输出取决于先前的计算。思考RNN的另一种方法是，它们具有“内存”，可以捕获有关到目前为止已计算出的内容的信息。理论上，RNN可以任意长的顺序使用信息，但实际上，它们仅限于回顾一些步骤。

·网络结构：

[RNN 结构详解](https://www.jiqizhixin.com/articles/2018-12-14-49)，循环神经网络的类型有很多，如下：

![RNN.png](https://i.loli.net/2020/05/25/nrhvYIotWKek1V2.png)

11.Long Short-term Memory, LSTM

·介绍：

长短期记忆(Long Short-Term Memory) （1997提出）是具有长期记忆能力的一种时间递归神经网络(Recurrent Neural Network)。 其网络结构含有一个或多个具有可遗忘和记忆功能的单元组成。它在1997年被提出用于解决传统RNN(Recurrent Neural Network) 的随时间反向传播中权重消失的问题（vanishing gradient problem over backpropagation-through-time)，重要组成部分包括Forget Gate, Input Gate, 和 Output Gate, 分别负责决定当前输入是否被采纳，是否被长期记忆以及决定在记忆中的输入是否在当前被输出。Gated Recurrent Unit 是 LSTM 众多版本中典型的一个。因为它具有记忆性的功能，LSTM经常被用在具有时间序列特性的数据和场景中。

LSTM网络由重复结构的LSTM单元组成，与RNN不同之处在于，重复的单元有四层特殊的结构（RNN只有一层）。

·网络结构：

[人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)、[长短期记忆网络](https://www.jiqizhixin.com/graph/technologies/e733c4c8-276b-428f-a43c-45e6044d5b7a)

![LSTM.png](https://i.loli.net/2020/05/25/yGdrxlfu7Q6LbcO.png)

12.GRU.

·介绍：

GRU（2014年提出）（Gate Recurrent Unit）是循环神经网络（Recurrent Neural Network, RNN）的一种。和LSTM（Long-Short Term Memory）一样，也是为了解决长期记忆和反向传播中的梯度等问题而提出来的。

GRU输入输出的结构与普通的RNN相似，其中的内部思想与LSTM相似。

相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

·网络结构：

[GRU的基本概念与原理](https://www.jiqizhixin.com/articles/2017-12-24)、[人人都能看懂的GRU](https://zhuanlan.zhihu.com/p/32481747)

<img src="https://i.loli.net/2020/05/25/d8OamlHYEFQcI1z.png" alt="GRU.png" style="zoom:80%;" />

13.Recursive Neural Network (RvNN) 

·介绍：

RvNN（2014提出的）的基本思想很简单：将处理问题在结构上分解为一系列相同的“单元”，单元的神经网络可以在结构上展开，且能沿展开方向传递信息。与RNN的思想类似，只不过将“时序”转换成了“结构”。

递归神经网络是一种深层神经网络，它是通过在结构化输入上递归应用相同的权重集，以在可变大小的输入结构上产生结构化预测来创建的，或对其进行标量预测，方法是按拓扑顺序遍历给定的结构。 RvNN在例如自然语言处理的学习序列和树结构（主要是基于词嵌入的短语和句子连续表示）方面已经取得了成功。

·网络结构：

[神经网络基础：DNN、CNN、RNN、RvNN、梯度下降、反向传播](https://blog.csdn.net/echoKangYL/article/details/86712657)、[Lecture 3 RvNN](https://www.jianshu.com/p/0249adb0b4c3)

<img src="https://i.loli.net/2020/05/25/nJtDKlxMwVirkIQ.png" alt="RvNN.png" style="zoom:80%;" />

14.Sequential CNN. 

·介绍：

Sequential CNN与使用RNN编码时间序列输入的常规工作不同，研究人员还研究了CNN以实现更有效的操作。借助递归神经网络顺序CNN，可以在训练过程中对所有元素的计算进行完全并行化，以更好地利用GPU硬件，并且由于非线性的数量是固定且与输入长度无关。

·网络结构：

![Sequentia CNN.jpg](https://i.loli.net/2020/05/25/esQd9tqm7zngBru.jpg)

15.External Memory Network.

·介绍：

外部存储网络。视觉注意力机制可以看作是一种短期记忆，可以将注意力分配到他们最近看到的输入功能上，而外部记忆网络可以通过读写操作来提供长期记忆。它已广泛用于许多应用中，例如视觉跟踪，视觉问答。

·网络结构：

![External Memory Network.jpg](https://i.loli.net/2020/05/25/Xi5rK1fUpV7d6bM.jpg)

16.Deep Generative Model.

·介绍：

深度生成模型基本都是以某种方式寻找并表达（多变量）数据的概率分布。有基于无向图模型（马尔可夫模型）的联合概率分布模型，另外就是基于有向图模型（贝叶斯模型）的条件概率分布。前者的模型是构建隐含层(latent)和显示层（visible)的联合概率，然后去采样。基于有向图的则是寻找latent和visible之间的条件概率分布，也就是给定一个随机采样的隐含层，模型可以生成数据。

近年来，深度生成模型取得了长足的发展，并且提出了许多流行的算法，例如VAE（可变自动编码器），GAN（生成对抗网络），CGAN（条件生成对抗网络）。可以找到这三个模型的图示。我们认为基于属性的行人图像生成策略可以处理低分辨率，不平衡的数据分布问题并显着增强训练数据集。

·网络结构：

[深度生成模型](https://www.jiqizhixin.com/graph/technologies/77c3d598-259e-4ecc-9617-bd27ca2098fd)、[Deep Generative Mode](https://www.cnblogs.com/Terrypython/p/9459584.html)

![Deep Generative Model.jpg](https://i.loli.net/2020/05/25/M65DIE9HJdynOAi.jpg)

#### 2.2.3.2 一些PAR算法<span id="2.2.3.2">

将从以下八个方面回顾基于深度神经网络的PAR算法：基于全局的，基于局部的，基于视觉注意的，基于顺序预测的，基于新设计的损失函数的，基于课程学习的，基于图形模型的以及其他算法。

1.基于全局的(Global Image-based Models)

在这一趴，将回顾仅考虑全局图像的PAR算法，比如ACN [5]，DeepSAR [6]，DeepMAR [6]，MTCNN [7]。

①ACN(ICCVW-2015)

ACN不考虑人体姿态，part及上下文信息，仅使用图像作为输入，训练CNN进行所有属性的预测。

![ACN-0.jpg](https://i.loli.net/2020/05/26/YEJjIuKi5xkRzbA.jpg)

ACN模型为卷积网络的每个属性学习提出了一个多分支分类层。如上图所示，采用经过预训练的AlexNet作为基本特征提取子网，并使用KL损失（基于Kullback-Leibler发散的损失函数）以每个属性一个损失替换最后一个完全连接的层。具体公式可描述如下：

$KL(P||Q)=\sum_{i}^{N}{P(Xi)log\frac{P(Xi)}{Q(Xi)}}$

$P(xi=yes)=l;p(xi=no)=1−l$

对于N/A的样本（提出了一个新的类别注释，即不可判定（N / A）。因为对于大多数输入图像，由于遮挡，图像边界或任何其他原因，某些属性无法确定），梯度设置为0。每个属性有一个损失函数，通过BP累积。最小化两个离散分布的KL 散度，Q是预测的结果，P是实际属性二值状态。目标空间是每个属性的交叉积，是一个结构化预测问题。

**数据库**
HATDB
Berkeley-行人属性数据集
PaRSE-27k数据集

**实验结果**

在HATDB数据集上与其他方法的对比

|                                                     | mAP                    |
| --------------------------------------------------- | ---------------------- |
| DSR [18]  SPM [13, 19]  EPM [19]  EPM + context[19] | 53.8  55.5  58.7  59.7 |
| *ACNH*  *ACNH* *5-ensemble*                         | **66.1**  **66.2**     |

在PARSE-27k数据集上的对比

![ACN-parse27.jpg](https://i.loli.net/2020/05/26/K1OZpwzMEqYjdA8.jpg)

在Berkeley行人属性数据集上与其他方法的对比

![ACN-Berkely.jpg](https://i.loli.net/2020/05/26/EZ8Tnv1C4PX6lGW.jpg)

[行人属性-ACN](https://blog.csdn.net/cv_family_z/article/details/78273634)、[论文] P . Sudowe, H. Spitzer, and B. Leibe, “Person attribute recognition with a jointly-trained holistic cnn model,” in Proceedings of the IEEE International Conference on Computer Vision Workshops, 2015, pp. 87–1, 3, 4, 5, 10, 11



②DeepSAR-DeepMAR(ACPR-2015)

[论文] D. Li, X. Chen, and K. Huang, “Multi-attribute learning for pedestrian attribute recognition in surveillance scenarios,” in Pattern Recognition(ACPR), 2015 3rd IAPR Asian Conference on. IEEE, 2015, pp. 111–1, 5, 10, 11, 26

该论文是后期被行人属性相关论文引用最多的。当前（2015年）属性识别问题主要针对两个应用场景，自然场景和监控场景。本篇论文针对监控场景。

该论文就行人属性识别领域存在的两个主要问题（手工找特征不能很好的适用视频场景、属性之间的关系被忽略），主要提出了两个网络，DeepSAR和DeepMAR。

DeepSAR：独立识别每个属性。将每一个属性的识别当作二元分类问题，然后一个一个识别每个属性。
DeepMAR：利用属性之间的关系，如长发更有可能是女性，所以头发的长度有利于帮助识别性别属性。将所有属性的识别一次性完成，多标签分类问题。

网络结构：

![DeepSAR-DeepMAR.png](https://i.loli.net/2020/05/28/iNjUZ5pfWYQ2dGF.png)

DeepSAR和DeepMAR共用ConvNet，其中ConvNet包括5个卷积层，3个全连接层。其后对应的激活单元 ReLU。前两个卷积层后面有Max Pooling层和Local Normalization层。最后一个卷积层后有Max Pooling层。模型在CaffeNet（CaffeNet和AlexNet基本一致，除了交换了归一化和池化的顺序）的基础上进行finetune。

DeepSAR的Loss function：

![DeepSAR-Loss.png](https://i.loli.net/2020/05/29/JuvB9yl2CEDMI1s.png)

 其中，N是行人图片的数量，L是属性的数量。pˆi,yil是第l个属性输出的softmax output probability。

DeepMAR的loss function：

![DeepMAR-Loss.png](https://i.loli.net/2020/05/29/cRZ25D7z8NLqpQx.png)

其中，wl是第l个属性的损失权重，pl是训练集中第l个属性的出现比例。σ是调优参数（设为1）

·实验结果

在PETA上的实验：总19000张——训练集:验证集:测试集 = 9500:1900:7600（PETA数据集常用分类方法）

![DeepSAR-PETA.png](https://i.loli.net/2020/05/29/qpOPBRYcUSKVrZv.png)

[DeepSAR-DeepMAR](https://blog.csdn.net/youshiwukong1524/article/details/83827533)



③MTCNN ((TMM-2015)

[论文] A. H. Abdulnabi, G. Wang, J. Lu, and K. Jia, “Multi-task cnn model for attribute prediction,” IEEE Transactions on Multimedia, vol. 17, no. 11,pp. 1949–1959, 2015. 1, 5, 10, 11

本文提出了一种使用CNN进行属性估计的联合多任务学习算法，称为MTCNN，如下图所示。

![MTCNN.jpg](https://i.loli.net/2020/05/29/ZOiIlbDUe4wTAfy.jpg)

MTCNN使CNN模型在不同属性类别之间共享视觉知识。

在CNN功能上采用多任务学习来估计相应的属性。在MTL框架中，他们还使用丰富的信息组，因为知道有关特征统计信息的任何先验信息肯定会帮助分类器。然后使用分解方法从总分类器权重矩阵W获得可共享的潜在任务矩阵L和组合矩阵S，从而通过学习局部特征（即W = LS）灵活地进行全局共享和组之间的竞争。因此，目标函数（称为MTL平方最大铰链损耗）公式如下：

min L，SMX m = 1 Nm X i = 1 1 2 [max（0,1-Yi m（Lsm）TXi m）] 2+ µ KX k = 1 GX g = 1 || sg k || 2+γ|| L || 1+λ|| L || 2 F

其中（Xi m，Yi m）Nm i = 1是训练数据中，Nmis是该属性的训练样本数。 K是潜在任务总维数空间。 第m个属性类别的模型参数表示为Lsm。采用加速近邻梯度下降（APG）算法以交替方式优化L和S。因此，可以在获得L和S之后获得整体模型权重矩W。



小结：

根据前面提到的算法，我们发现这些算法都将整个图像作为输入，并对PAR进行多任务学习。他们都尝试通过特征共享，端到端训练或多任务学习平方最大铰链损耗来学习更强大的特征表示。这些模型的好处是简单，直观和高效，这对于实际应用非常重要。但是，由于缺乏细粒度识别的考虑，这些模型的性能仍然受到限制。



2.基于局部的(Part-based Models)

在这一趴中，将整理介绍基于局部的算法，这些算法可以联合利用本地和全局信息来获得更准确的PAR。这些算法包括：Poselets [8]，RAD [9]，PANDA [10]，MLCNN [11]，AAWP [12]，ARAP [13]，DeepCAMP [14]，PGDM [15]，DHC [16]， LGNet [17]。

[8] L. Bourdev, S. Maji, and J. Malik, “Describing people: A poselet-based approach to attribute classification,” in Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011, pp. 1543–1550. 1, 3,4, 5, 11, 12, 13, 15
[9] J. Joo, S. Wang, and S.-C. Zhu, “Human attribute recognition by rich appearance dictionary,” in Proceedings of the IEEE International Conference on Computer Vision, 2013, pp. 721–728. 1, 5, 11, 15
[10] N. Zhang, M. Paluri, M. Ranzato, T. Darrell, and L. Bourdev, “Panda:Pose aligned networks for deep attribute modeling,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014,pp. 1637–1644. 1, 5, 11, 12, 13, 15, 26
[11] J. Zhu, S. Liao, D. Yi, Z. Lei, and S. Z. Li, “Multi-label cnn based pedestrian attribute learning for soft biometrics,” in Biometrics (ICB), 2015 International Conference on. IEEE, 2015, pp. 535–540. 1, 5, 11,12, 15
[12] G. Gkioxari, R. Girshick, and J. Malik, “Actions and attributes from wholes and parts,” in The IEEE International Conference on Computer Vision (ICCV), December 2015. 1, 11, 13, 15
[13] Y . W. S. L. Luwei Yang, Ligeng Zhu and P . Tan, “Attribute recognition from adaptive parts,” in Proceedings of the British Machine Vision Conference (BMVC), E. R. H. Richard C. Wilson and W. A. P .
Smith, Eds. BMV A Press, September 2016, pp. 81.1–81.11. [Online].Available: https://dx.doi.org/10.5244/C.30.81 1, 11, 13, 15
[14] A. Diba, A. Mohammad Pazandeh, H. Pirsiavash, and L. V an Gool, “Deepcamp: Deep convolutional action & attribute mid-level patterns,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 3557–3565. 1, 11, 14, 15
[15] D. Li, X. Chen, Z. Zhang, and K. Huang, “Pose guided deep model for pedestrian attribute recognition in surveillance scenarios,” in 2018 IEEE International Conference on Multimedia and Expo (ICME).  IEEE,
2018, pp. 1–6. 1, 11, 14, 15
[16] Y . Li, C. Huang, C. C. Loy, and X. Tang, “Human attribute recognition by deep hierarchical contexts,” in European Conference on Computer Vision. Springer, 2016, pp. 684–700. 1, 2, 3, 4, 11, 14, 15
[17] P . Liu, X. Liu, J. Yan, and J. Shao, “Localization guided learning for pedestrian attribute recognition,” 2018. 1, 11, 14, 15

①Poselets (ICCV-2011)

Poselets 的动机是，如果我们可以从相同的角度隔离对应于相同身体部位的图像块，则可以更简单地训练属性分类器。但是，由于当时的能力有限，当时（2011年之前）直接使用物体检测器对于身体部位的定位并不可靠。因此，作者采用了小波将图像分解为一组部分，每个部分都捕获了与给定视点和局部姿势相对应的显着模式。如下图所示：



![Poselets.jpg](https://i.loli.net/2020/05/29/mF5YjJPkGZWyv2X.jpg)

具体而言，首先检测给定图像上的姿势，然后通过将HOG，颜色直方图和皮肤蒙版特征进行级联来获得其关节表示。然后训练了多个SVM分类器，分别用于姿势集级，人员级，上下文级属性分类。

姿势级分类器的目标是在给定视角下从人的给定部位确定属性的存在。

人员级别分类器用于组合来自身体各个部位的证据，而上下文级别分类器则将所有人员级别分类器的输出作为输入，并尝试利用属性之间的相关性。

它们的属性预测结果是上下文级别分类器的输出。

②RAD (ICCV-2013)

RAD 从外观变化的角度提出了一种零件学习算法，而先前的工作则集中在处理需要人工标注零件的几何变化上，例如Poselets。

首先将图像晶格划分为多个重叠的子区域（称为窗口）。如下图（a）所示，定义了尺寸为W×H的网格，并且网格上包含一个或多个单元格的任何矩形形成一个窗口。所提出的方法在零件窗口的形状，大小和位置上具有更大的灵活性，而先前的工作（例如空间金字塔匹配结构，SPM ）将区域递归地划分为四个象限，并使所有子区域是不与每个区域重叠的正方形其他处于同一水平。通过所有这些窗口，他们可以学习与该特定窗口在空间上关联的一组零件检测器。对于每个窗口，从训练图像中裁剪所有对应的图像块，并由HOG 和颜色直方图特征描述符表示。然后，基于提取的特征进行K均值聚类。每个获得的簇表示零件的特定外观类型。他们还通过逻辑回归为每个聚类训练一个局部检测器，作为初始检测器，并通过将其再次应用于整个集合并更新最佳位置和比例来处理嘈杂的聚类问题，从而对其进行迭代优化。

![RAD.jpg](https://i.loli.net/2020/05/29/68jKGoNl7IgB9zq.jpg)



在多尺度重叠窗口中学习零件后，他们遵循基于Poselet的方法中提出的属性分类方法。具体而言，它们将来自这些局部分类器的分数与由零件检测分数给出的权重进行汇总，以进行最终预测。

③PANDA (CVPR-2014)

张等发现与某些属性相关的信号是微妙的，并且图像受姿势和视点的影响所支配。对于戴眼镜的属性，信号在整个人的范围内都很微弱，并且外观随头部姿势，框架设计和头发的遮挡而显着变化。他们认为准确预测基础属性的关键在于定位对象部分并建立它们与模型部分的对应关系。他们建议共同使用全局图像和局部补丁进行人员属性识别，整个流程可以在下图（a）和（b）中找到。

<img src="https://i.loli.net/2020/05/29/PdaTMKbCn5oYsZh.jpg" alt="PANDA.jpg"  />

如上图(a)所示，他们首先检测姿势并获得人的部分。然后，他们采用CNN提取局部补丁和整个人类图像的特征表示。对于未检测到的姿势，他们只需将特征保持为零即可。因此，他们的模型既可以利用卷积网络的能力来从数据中学习判别特征，又可以利用小波集的能力通过将对象分解为典型的姿势来简化学习任务。他们将组合的局部和全局特征直接输入到线性分类器中，该线性分类器是用于多属性估计的SVM（支持向量机）。上图展示了详细的体系结构，该网络以体态RGB补丁56×56×3作为输入，输出具有相应的完全连接层（fc层）的每个属性的响应得分。

特征提取模块包含四组卷积/池化/归一化层。用于深度属性建模（PANDA）的姿势对齐网络概述。在每个姿势的语义部分补丁上训练一个卷积神经网络，然后将所有网络的顶层激活连接起来以获得姿势归一化的深度表示。线性SVM分类器使用姿势归一化表示来预测最终属性。这个数字来自PANDA。这些组的输出分别是28×28×64、12×12×64、6×6×64和3×3×64。然后，将输入图像映射到尺寸为576-D的具有完全连接层的特征向量。他们为输出尺寸为128的每个属性设置了一个fc层。

为了这项工作的优势，它采用深层特征而不是浅层低层特征，与以前的工作相比，浅层低层特征可以获得更强大的特征表示。此外，它还从局部补丁和全局图像的角度处理人的图像，与仅考虑整个图像的作品相比，它可以挖掘更详细的信息。这两点都大大改善了人的属性识别。但是，我们认为以下问题可能会限制程序的最终性能：1）。零件的定位，即姿势的准确性，可能是其结果的瓶颈； 2）。他们没有使用端到端的学习框架来学习深度特征； 3）。它们的姿势还包含背景信息，这也可能会影响特征表示。

④MLCNN(ICB-2015) 

本文提出了一种多标签卷积神经网络，可以在一个统一的框架中一起预测多个属性。他们的网络的整体管线可以在下图中找到。

![MLCNN.jpg](https://i.loli.net/2020/05/30/6olRSyb7CJtzNei.jpg)

他们将整个图像分为15个重叠的小块，并使用卷积网络提取其深层特征。他们采用相应的局部进行特定的属性分类，例如将patch1,2,3用于发型评估。他们为每个属性预测使用softmax函数。

此外，他们还使用预测的属性来协助人员重新识别。具体来说，他们将低级特征距离和基于属性的距离融合为最终融合距离，以区分给定图像是否具有相同的身份。

⑤AAWP(ICCV-2015)

引入AAWP是为了验证零件是否可以改善动作和属性识别。如下图（1）所示，在与实例相关联的一组边界框上计算CNN特征，以对整个实例进行分类，即对整个实例进行分类，即提供的先知或人物检测器以及提供的姿势样部件检测器。作者定义了三个人体部位（头部，躯干和腿），并将每个部位的要点聚集到几个不同的姿势中。由于利用了深部特征金字塔，而不是传统体位中使用的低水平梯度方向特征，该部分检测器被称为体位的深层版本。另外，作者还介绍了特定于任务的CNN微调，他们的实验表明，微调的整体模型（即没有零件）已经可以与基于零件的系统（如PANDA）取得可比的性能。具体地，整个管线可以分为两个主要模块，即部分检测器模块和细粒度分类模块，分别如下图（2）和（3）所示。

![AAWP.jpg](https://i.loli.net/2020/05/30/bJmZT67foluOsrS.jpg)

对于零件检测器模块，他们通过遵循对象检测算法RCNN 来设计其网络，该算法包含两个阶段，即特征提取和零件分类。他们采用了多尺度全卷积网络来提取图像特征。更具体地说，他们首先构造彩色图像金字塔并获得每个金字塔级别的pool5特征。然后，他们采用零件模型来获得相应的分数，如上图（2）所示。因此，关键问题在于在给定这些特征图金字塔的情况下如何实现精确的零件定位。为了处理零件的定位，作者设计了三个身体区域（头部，躯干和腿），并使用线性SVM训练零件检测器。积极的训练数据是从PASCAL VOC 2012收集的，采用了聪明的算法。在测试阶段，它们将得分最高的部分保留在图像的候选区域框中。对于他们的论文中讨论的基于零件的分类任务，即动作和属性识别。他们考虑了四种不同的方法来了解哪些设计因素很重要，即没有零件，例如微调，联合微调和三向拆分。上图（3）中提供了用于细粒度分类的详细管道。给定图像和检测到的部分，他们使用CNN获得fc7特征并将其并入一个特征向量作为其最终表示。因此，可以使用预训练的线性SVM分类器来估计动作或属性类别。他们在PASCAL VOC行为挑战和人数据集的Berkeley属性上进行的实验[8]验证了零件的有效性。此外，他们还发现，随着设计更强大的卷积网络体系结构，显性部件的边际收益可能消失。他们认为这可能是由于整体网络已经获得了高性能。这项工作以更广泛的方式进一步扩展并验证了零件的有效性和必要性。它还显示了有关基于深度学习的人的属性识别的更多见解。

 ⑥ARAP(BMVC-2016) 

本文采用端到端的学习框架进行关节局部定位和人属性识别的多标签分类。如下图所示，ARAP包含以下子模块：初始卷积特征提取层，关键点定位网络，每个部分的自适应边界框生成器以及每个部分的最终属性分类网络。它们的网络包含三个损失函数，即回归损失，纵横比损失和分类损失。

具体来说，他们首先提取输入图像的特征图，然后进行关键点定位。根据关键点，他们将人体分为三个主要区域（包括坚硬，躯干和腿部），并获得初始零件边界框。另一方面，他们也将以前的fc7图层的要素用作输入，并估计边界框调整参数。给定这些边界框，他们采用双线性采样器提取相应的局部特征。然后，将要素馈入两个fc层以进行多标签分类。

![ARAP.jpg](https://i.loli.net/2020/05/31/D32dQbhOayYtAvV.jpg)

⑦DeepCAMP(ICCV-2016) 

本文提出了一种新颖的CNN，它可以挖掘中级图像补丁来进行细粒度的人类属性识别。具体来说，他们训练CNN来学习具有区别性的补丁程序组，称为DeepPattern。他们利用常规的上下文信息（参见下图（2）），还让特征学习和补丁聚类的迭代纯化了专用补丁集，如下图（1）所示。本文的主要见解在于，更好的嵌入可以帮助提高模式挖掘算法中聚类算法的质量。因此，他们提出了一种迭代算法，其中在每次迭代中，他们训练一个新的CNN来对在先前迭代中获得的聚类标签进行分类，以帮助改善嵌入。另一方面，它们还将本地补丁和全局人类边界框的特征连接在一起，以改善中级元素的群集。

![DeepCAMP.jpg](https://i.loli.net/2020/05/31/2I4wj9pJHaPVkts.jpg)

⑧PGDM (ICME-2018)

PGDM是尝试探索行人身体的结构知识（即行人姿势）以进行人的属性学习的第一项工作。他们首先使用预先训练的姿势估计模型估计给定人类图像的关键点。然后，他们根据这些关键点提取零件区域。局部区域和整个图像的深层特征都被提取出来，并独立用于属性识别。然后将这两个分数融合在一起以实现最终的属性识别。姿态估计的可视化和PGDM的整个流水线分别可见于下图（a）和（b）。

如下图（b）所示，属性识别算法包含两个主要模块：即主网和PGDM。主网是对AlexNet的修改，fc8层设置为与属性编号相同。它把属性识别作为多标签分类问题，并采用改进的交叉熵损失作为目标函数。对于PGDM模块，其目标是探索可变形的身体结构知识，以辅助行人属性识别。作者诉诸于深层姿势估计模型，而不是重新标注训练数据中的人类姿势信息。他们将现有的姿态估计算法嵌入到其属性识别模型中，而不是将其用作外部模型。他们直接训练一个回归网络，以使用从现有姿态估计模型获得的粗糙地面真实姿态信息来预测行人姿态。一旦获得姿势信息，他们就可以使用空间变换器网络（STN）将关键点转换为信息丰富的区域。然后，他们使用独立的神经网络从每个关键点相关区域进行特征学习。他们共同优化了主网络，PGDM和姿态回归网络。

![PGDM.jpg](https://i.loli.net/2020/05/31/831pKQyOjmICZeP.jpg)

⑨DHC (ECCV-2016)

由于背景有时会提供比目标对象更多的信息，因此本文提出使用深层次的上下文来帮助识别人的属性。尤其是，在他们的网络体系结构中引入了以人为中心的上下文和场景上下文。

如下图所示，他们首先构造输入图像金字塔，然后将它们全部通过CNN（本文使用VGG-16网络），以获得多尺度特征图。他们提取四组边界框区域的特征，即整个人，目标对象的检测部分，图像金字塔和全局图像场景中的最近邻居部分。前两个分支（整个人员和部分）是用于人员属性识别算法的常规管道。本文的主要贡献在于后两个分支，即以人为中心和场景级别的上下文，以帮助改善识别结果。一旦获得了这四个分支的分数，它们会将所有分数汇总为最终属性分数。由于使用了上下文信息，因此该神经网络比常规的行人属性识别任务需要更多的外部训练数据。例如，他们需要检测人体的一部分（头部，上半身和下半身区域）并识别给定图像的样式/场景。他们提出了一个名为WIDER的新数据集，以更好地验证他们的想法。尽管可以通过此管道显着改善人的属性识别结果，但是，该模型看起来比其他算法复杂。

![DHC.jpg](https://i.loli.net/2020/05/31/bYWcdxXGAgsQuSM.jpg)

⑩LGNet (2018)

本文提出了一个Localization Guide Network（名为LGNet），它可以对与不同属性相对应的区域进行本地化，也遵循局部全局框架，如下图所示。

具体地说，他们采用Inception-v2作为特征提取的基本CNN模型。对于全球分支机构，他们采用全球平均池化层（GAP）以获得其全局功能。然后，使用完全连接的层来输出其属性预测。对于本地分支，使用1×1卷积层为每个图像生成c类激活图，其中c是所用数据集中的属性数。给定类激活图，他们可以通过裁剪相应激活图的高响应区域来捕获每个属性的激活框。他们还使用EdgeBoxes生成区域建议，以从输入图像中获取局部特征。此外，他们还考虑了提取提案的不同贡献，并且不同属性应关注不同的本地特征。因此，他们使用每个属性的类活动地图来作为确定局部特征对不同属性的重要性的指南。更具体地说，他们根据联合互动（IoU）计算提案和类别激活框之间的空间亲和度地图，并对其进行线性归一化以加权局部特征向量以进行进一步的预测。最后，全局和有人参与的局部特征通过逐元素总和融合在一起，用于行人属性预测。

![LGNet.jpg](https://i.loli.net/2020/05/31/qtDhnc9L26PmQwH.jpg)

小结：

根据以上提到的算法，很容易发现这些算法都采用联合利用全局和细粒度的局部特征。身体部位的定位是通过外部部位定位模块来实现的，例如，部位检测，姿势估计，姿势集或提议生成算法。使用零件信息可以显着提高整体识别性能。同时，它也带来一些不足之处：首先，作为中间阶段的操作，最终的识别性能在很大程度上取决于part localization。换句话说，不正确的零件检测结果将为最终分类带来错误的特征。其次，由于引入了人体部位，因此还需要更多的训练或推理时间。第三，一些算法需要关于零件位置的手动注释标签，这进一步增加了人力和金钱成本。

3.基于视觉注意的(Attention-based Models)

在这一趴中，将整理介绍使用注意力机制的人属性识别算法，例如HydraPlus-Net [18]，VeSPA [19]，DIAA [20]，CAM [21]。

[18] X. Liu, H. Zhao, M. Tian, L. Sheng, J. Shao, S. Yi, J. Yan, and X. Wang, “Hydraplus-net: Attentive deep features for pedestrian analysis,” in Proceedings of the IEEE International Conference on Computer Vision,
2017, pp. 350–359. 1, 2, 3, 4, 15, 16, 26

[19] M. S. Sarfraz, A. Schumann, Y . Wang, and R. Stiefelhagen, “Deep view-sensitive pedestrian attribute inference in an end-to-end model,” arXiv preprint arXiv:1707.06089, 2017. 1, 15, 16

[20] N. Sarafianos, X. Xu, and I. A. Kakadiaris, “Deep imbalanced attribute classification using visual attention aggregation,” in European Conference on Computer Vision.  Springer, 2018, pp. 708–725. 1, 15, 16,26

[21] H. Guo, X. Fan, and S. Wang, “Human attribute recognition by refining attention heat map,” Pattern Recognition Letters, vol. 94, pp. 38–45,2017.1, 15, 17



①HydraPlus-Net（ICCV-2017）

HP-Net是一个基于注意力机制（attention-based）的深度神经网络，将多层注意力机制图多向映射到不同的特征层。可以使用多方向注意对来自多个级别的多尺度特征进行编码，以进行行人分析，这是（MDA）模块。

如下图所示，它包含两个主要模块，即作为常规CNN的主网（M-net）和包括应用的多方向注意模块的多个分支的Attentive Feature Net（AF-net）到不同的语义特征级别。 AF-net和M-net共享相同的基本卷积体系结构，它们的输出由全局平均池（GAP）和fc层连接和融合。输出层可以是用于属性识别的属性logit或用于人员重新识别的特征向量。作者采用inceptionv2作为他们的基本网络。

![HP-Net.jpg](https://i.loli.net/2020/06/01/JvMzSsGgNnpxWA1.jpg)

AF-net的特定图示可以在上图中找到。给定黑色1、2和3的特征图，它们对特征图2进行1×1卷积运算并获得其注意图α2。值得注意的是，该关注模块与以前的基于关注的模型不同，后者仅将关注图推回同一块。他们不仅使用此关注图来参与要素图2，而且还使用相邻的要素（例如要素图1和3）。将一个关注图应用于多个块自然可以使融合的要素在同一空间内对多级信息进行编码分布，如上图所示。 HP-net是按阶段进行训练的，换句话说，M-net，AF-net和其余的GAP和fc层是按顺序训练的。输出层用于最小化交叉熵损失和softmax损失，分别用于人员属性识别和人员重新识别。

实验结果：

![HP-Net-result.png](https://i.loli.net/2020/06/01/NskdZgWGTjwUDK1.png)

[原文地址](https://arxiv.org/abs/1709.09930)、[Github]( https://github.com/xh-liu/HydraPlus-Net)、[论文阅读](https://www.cnblogs.com/White-xzx/p/10203964.html)

②VeSPA （arXiv-2017）

VeSPA考虑了视图提示，以更好地估计相应的属性。因为作者发现提示属性的视觉提示可以很强地定位，而对人的属性（如头发，背包，短裤等）的推断高度依赖于行人的视野。如下图所示，图像被送入Inceptions（K层）并获得其特征表示。引入特定于视图的单元以将特征图映射到粗略属性预测y_att = [y1，y2，...，yc] T。然后，使用视图预测器来估计视图权重y yview。注意权重用于乘以特定于视图的预测，并获得最终的多类属性预测attribute_Yc = [y1，y2，...，yC] T。使用单独的损失函数训练视图分类器和属性预测器。

![VeSPA.jpg](https://i.loli.net/2020/06/01/NgnGcwFpEld4C5e.jpg)

③DIAA （ECCV-2018）

DIAA算法可以看作是用于人的属性识别的整体方法。如下图所示，他们的模型包含以下模块：多尺度视觉注意力和加权失焦，用于深度不平衡分类。从下图可以看出，对于多尺度视觉注意力，作者采用了来自不同层的特征图。他们提出了加权焦点损失函数来测量预测属性向量和地面真实性之间的差异。此外，他们还建议以弱监督的方式（仅属性标签，没有特定的边界框注释）学习注意力图，以通过指导网络将其资源集中到包含与以下内容有关的信息的空间部分来提高分类性能去输入图像。

![DIAA.jpg](https://i.loli.net/2020/06/01/3o5Ihd7Hjq2lQGN.jpg)

如上图的右侧所示，注意子网络以特征图Wi×Hi×Fias输入并输出尺寸为Wi×Hi×C的注意掩码。然后输出到注意力分类器以估计行人属性。由于用于监督注意力模块训练的监督信息有限，因此作者诉诸预测方差。跨时间具有高标准偏差的注意掩码预测将被赋予更高的权重，以引导网络学习那些不确定的样本。他们收集了sample的预测历史H，并计算了小批量中每个样本的时间跨度标准差。因此，可以通过以下公式获得每个样本s的带有属性级别监督的注意力图的损失：Lai（ˆ yai，y）=（1 + stds（H））Lb（ˆ yai，y）（18）其中Lb （ˆ yai，y）是二元交叉熵损失，stds（H）是标准偏差。因此，用于端到端训练此网络的总损失是主网络和两个注意模块的损失之和。

实验结果：

①result in wider attribution

![DIAA-Wider.png](https://i.loli.net/2020/06/01/ufdCVak4EL3X2rj.png)

②result in PETA

![DIAA-PETA.png](https://i.loli.net/2020/06/01/u4szdgTEWVf1tmi.png)

④CAM（PRL-2017）

在本文中，作者提议使用和完善注意力图来提高人属性识别的性能。如下图所示，它们的模型包含两个主要模块，即多标签分类子网和关注地图提炼模块。所采用的CAM网络也遵循类别特定的框架，换句话说，对于完全连接（FC）层，不同的属性分类器具有不同的参数。他们使用FC层中的参数作为权重，以线性组合最后一个卷积层中的特征图，以获得每个对象类别的注意图。

但是，由于分辨率低，训练过度等原因，这种幼稚的注意力机制实施方法无法始终将注意力集中在正确的区域上。为了解决上述问题，他们尝试通过调整CAM网络来完善注意力图。他们根据注意力的集中度来衡量注意力图的适当性，并尝试使注意力图突出显示一个较小但集中的区域。具体来说，他们引入了加权平均层来首先获得注意力图。然后，他们使用平均池对分辨率进行下采样，以捕获所有潜在相关区域的重要性。

之后，他们还采用Softmax层将注意力图转换为概率图。最后，可以通过全局平均池化层获得最大概率。基于最大概率，作者提出了一个新的损失函数（称为指数损失函数）来衡量关注热点图的适当性，可以写成：L = 1Neα（PM ij + βµ）其中PM ij是图像i和属性j的最大概率。 α和β是超参数，μ= 1 / H2是概率图的平均值。 H×H是注意（和概率）图的大小。对于网络的训练，作者首先仅通过最小化分类损失来对CAM网络进行预训练。然后，他们采用联合损失功能来微调整个网络。

![CAM.jpg](https://i.loli.net/2020/06/01/tk5D3MHTC1SJPgz.jpg)

小结：

视觉注意机制已经在行人属性识别中引入，但是现有的工作仍然有限。在这一领域，仍然需要探索如何设计新的注意力模型或直接从其他领域借鉴。



4.基于顺序预测的(Sequential Prediction based Models)

在这一趴中，我们将介绍基于顺序预测的人属性识别模型，包括CNN-RNN [22]，JRL [23]，GRL [24]，JCM [25]和RCRA [101]。

[22] J. Wang, Y . Yang, J. Mao, Z. Huang, C. Huang, and W. Xu, “Cnn-rnn: A unified framework for multi-label image classification,” in Proceedings of the IEEE conference on computer vision and pattern recognition,
2016, pp. 2285–2294. 1, 17, 18
[23] J. Wang, X. Zhu, S. Gong, and W. Li, “Attribute recognition by joint recurrent learning of context and correlation,” in Computer Vision(ICCV), 2017 IEEE International Conference on.  IEEE, 2017, pp.
531–540. 1, 17, 18
[24] X. Zhao, L. Sang, G. Ding, Y . Guo, and X. Jin, “Grouping attribute recognition for pedestrian with joint recurrent learning.” in IJCAI, 2018, pp. 3177–3183. 1, 17, 18, 19
[25] H. Liu, J. Wu, J. Jiang, M. Qi, and R. Bo, “Sequence-based person attribute recognition with joint ctc-attention model,” arXiv preprint arXiv:1811.08115, 2018. 1, 17, 18, 19, 28

[101] G. D. J. H. N. D. Xin Zhao, Liufang Sang and C. Yan, “Recurrent attention model for pedestrian attribute recognition ,” in Association for the Advancement of Artificial Intelligence, AAAI, 2019. 17, 19

①CNN-RNN (CVPR-2016)

常规的多标签图像分类框架为每个类别学习独立的分类器，并对分类结果采用排名或阈值，无法显式利用图像中的标签依赖性。本文首先采用RNN来解决此问题，并与CNN结合学习联合的图像标签嵌入，以表征语义标签的依赖关系以及图像标签的相关性。

如下图所示，红色和蓝色点分别是标签和图像嵌入。对图像和递归神经输出嵌入进行求和，并用黑点表示。该机制可以通过顺序链接标签嵌入来对联合嵌入空间中的标签共现依赖性进行建模。它可以根据图像嵌入I和递归神经元xt的输出来计算标签的概率，可以将其表达为：s（t）= UT lxt（20）其中xt = h（Ux oo（t）+ Ux II ），Ux o和Ux I分别是输出和图像表示的递归层的投影矩阵。 Ulis标签嵌入矩阵。 o（t）是时间步t处递归层的输出。为了推论CNN-RNN模型，他们尝试找到最大化先验概率的标签序列：l1，...，lk = arg max l1，...，lkP（l1，...，lk | I ）= arg max l1，...，lkP（l1 | I）×P（l2 | I，l1）... P（lk | I，l1，...，lk-1）他们采用梁排名靠前的预测路径的搜索算法作为其估计结果。可以通过交叉熵损失函数和时间反向传播（BPTT）算法来实现CNN-RNN模型的训练。

![CNN-RNN.jpg](https://i.loli.net/2020/06/01/9Jmk61TYKnpB4Vb.jpg)

 CNN-RNN模型与用于图像字幕任务的深层模型非常相似。它们都以一个图像作为输入，并在编码器-解码器框架下输出一系列单词。主要区别在于字幕模型输出一个句子，而CNN-RNN模型生成属性（但这些属性也相互关联）。因此，我们可以借鉴图像标题社区的一些技术，以帮助提高行人属性识别的性能。

 ②JRL (ICCV-2017)

本文首先分析行人属性识别任务中存在的学习问题，例如图像质量差，外观变化少且带注释的数据少。 他们建议探索属性与视觉环境之间的相互依赖性和相关性作为辅助属性识别的额外信息源。 因此，提出JRL模型以联合属性的递归学习上下文和相关性，其的整体渠道JRL可以如下图中所示。

为了更好地挖掘这些额外的信息以进行准确的人物属性识别，作者采用序列到序列模型来处理上述问题。它们首先将给定的人物图像I分成m个水平条带区域，并形成区域序列S = (s1，s2，...sm)以自上而下的顺序排列。所获得的区域序列S可以被视为自然语言处理中的输入语句，并且可以用LSTM网络以顺序的方式进行编码。编码器LSTM的隐藏状态表可以基于常规的LSTM更新过程来更新。当MCA可以被看作是整个人图像的概要表示z = MOF时的最终隐藏状态(称为上下文向量)。该特征提取过程可以对每个人图像中的人内属性上下文进行建模。为了挖掘更多的辅助信息来处理目标图像中的外观模糊和图像质量差。作者借助视觉上相似的样本训练图像，并引入这些样本来建模人与人之间的相似性上下文约束。首先基于L2距离度量搜索与目标图像相似的具有CNN特征的top-k样本，并计算其自身的上下文向量。然后，所有的上下文向量表示被集合为具有最大汇集操作的人与人之间的上下文z *。在解码阶段，解码器LSTM将个人内属性上下文(z)和个人间相似性上下文(z∑)作为输入和输出可变长度属性。本文中的属性预测也可以看作是一种生成方案。为了更好地关注人物图像中特定属性的局部区域并获得更准确的表示，他们还引入了注意机制来关注人物内部的属性上下文。对于最终的属性估计顺序，他们采用集成的思想来整合不同顺序的互补优势，从而在上下文中捕获属性之间更高阶的相关性。

![JRL.jpg](https://i.loli.net/2020/06/01/IzkW7nwNi4AcTZQ.jpg)

③GRL (IJCAI-2018)

GRL是在JRL的基础上发展起来的，后者也采用了RNN模型来顺序预测人的属性。与JRL不同的是，GRL通过一步一步的分组来识别人的属性，以关注组内和组间的关系。如下图 (1)所示，作者将整个属性列表分成许多组，因为组内的属性是互斥的，并且组间有关系。例如，黑头发和黑头发不能出现在同一个人物图像上，但是它们都与人物的头肩区域相关，并且可以在同一组中一起被识别。这是一种端到端的单模型算法，不需要预处理，并且在分组的行人属性中利用了更多潜在的组内和组间相关性。

整个算法可以在下图 (2)中找到。如下图 (2)所示，给定人类图像，他们首先检测关键点，并使用身体区域生成模块定位头部、上身和下身区域。它们利用Inception-v3网络提取整个图像的特征，并利用投资回报率平均池操作获得局部特征。值得注意的是，同一组中的所有属性共享相同的完全连接的特征。在给定全局和局部特征的情况下，他们采用LSTM对属性组中的空间和语义相关性进行建模。每个LSTM单元的输出然后被馈送到完全连接的层，并且可以获得预测向量。该向量的维数与相关组中的属性数相同。他们还使用一个批处理规范化层来平衡这个网络的正负输出。

![GRL.jpg](https://i.loli.net/2020/06/01/w7pNmxB2FoPajcV.jpg)

④JCM (arXiv-2018)

现有的基于序列预测的行人属性识别算法，例如JRL、GRL，由于RNN的弱对齐能力，可能容易受到不同的手动划分和属性顺序的影响。本文提出了一种联合关注度模型(JCM)来进行属性识别，该模型可以一次预测任意长度的多个属性值，避免了映射表中属性顺序的影响。

如下图所示，JCM实际上是一个多任务网络，它包含两个任务，即属性识别和人的再识别。他们使用ResNet-50作为基本模型来提取这两项任务的特征。对于属性识别，他们采用变压器作为长属性序列比对的注意模型。在网络训练中，采用了连接主义时间分类损失函数和交叉熵损失函数。对于重新识别的人，他们直接使用两个完全连接的层(即密集模型)获取特征向量，并使用软最大损失函数来优化该分支。在测试阶段，JCM可以同时预测人的身份和一组属性。他们还使用波束搜索来解码属性序列。同时，在基本模型中提取网络特征，对行人进行分类，完成身份识别任务。

![JCM.jpg](https://i.loli.net/2020/06/01/ARWSbGa82tCBdKq.jpg)

⑤RCRA (AAAI-2019)

本文提出了两种行人属性识别模型，即递归卷积模型和递归注意模型，如下图所示。RC模型利用卷积LSTM模型来探索不同属性组之间的相关性，而RA模型利用组内空间局部性和组间注意相关性来提高最终性能。具体来说，他们首先将所有属性分成多个属性组，类似于GRL。对于每一幅行人图像，他们使用美国有线电视新闻网提取其特征地图，并逐组提供给康沃尔斯特姆图层。然后，通过在卷积后增加一个卷积网络，可以得到每个时间步长的新的特征映射。最后，将特征用于当前属性组的属性分类。基于上述RC模型，他们还引入了视觉注意模块来突出特征地图上的感兴趣区域。给定在每个时间步骤t的关注的图像特征图f和热图htf，可以通过以下方式获得当前属性组的关注特征图ftf:ft = sigmoid(ht)⊗f+f(22 ),其中⊗表示空间逐点乘法。关注的特征图用于最终分类。该网络的训练也基于在WPAL-网络中提出的加权交叉熵损失函数。

![RCRA.jpg](https://i.loli.net/2020/06/01/2swRI9FkzMCPryN.jpg)



小结：

从这一小节我们可以看出，这些算法都采用了序列估计过程。因为属性是相互关联的，它们也有各种各样的困难。因此，采用RNN模型逐个估计属性是一个有趣而直观的想法。在这些算法中，它们将不同的神经网络、属性组、多任务学习集成到这个框架中。与基于有线电视新闻网的方法相比，这些算法更优雅和有效。这些算法的缺点是连续属性估计的时间效率。在未来的工作中，需要更有效的序列属性估计算法。



5.基于新设计的损失函数的(Loss Function based Models)

在这一趴中，将整理一些改进损失函数的算法，包括WPAL[26]，AWMT [27]。

[26] Y . Zhou, K. Y u, B. Leng, Z. Zhang, D. Li, K. Huang, B. Feng, C. Yao et al., “Weakly-supervised learning of mid-level features for pedestrian attribute recognition and localization,” in BMVC, 2017. 1, 19, 20, 26,27
[27] K. He, Z. Wang, Y . Fu, R. Feng, Y .-G. Jiang, and X. Xue, “Adaptively weighted multi-task deep network for person attribute classification,” in Proceedings of the 2017 ACM on Multimedia Conference.  ACM, 2017, pp. 1636–1644. 1, 19, 20, 21, 27

①WPAL-network (BMVC-2017)

WPAL被提议以弱监督的方式同时识别和定位人的属性(即仅人的属性标签，没有特定的边界框注释)。如下图所示，采用谷歌网络作为其特征提取的基本网络。

它们融合来自不同层的特征(即，来自Conv3、Conv2和Conv1层的特征)，并馈入灵活的空间金字塔汇集层(FSPP)。与常规的全局最大池相比，FSPP的优势可以归纳为以下两个方面:1).它可以为帽子等属性添加空间约束；2).该结构位于网络的中间阶段，而不是顶层，使得检测器和目标类之间的相关性起初不受限制，但在训练过程中可以自由学习。每个FSPP的输出被馈送到完全连接的层中，并输出其维数与行人属性的数量相同的向量。在训练过程中，网络可以同时学习适应以下两个目标:第一个是学习属性与随机初始化的中层检测器之间的相关性，第二个是调整检测器的目标中层特征以适应相关属性。学习到的相关性、中层特征的检测结果随后被用于定位人的属性。此外，作者还引入了一种新的加权交叉熵损失函数来处理大多数属性类别的正负样本的极不平衡分布。数学公式可以写成如下:Losswce = L×I = 1 1 2 wi∫pi∫log(π)+1 2(1 wi)(1π)∫log(1π)，其中L表示属性的数量，p是地面真实属性向量，p是估计属性向量，w是表示训练数据集中所有属性类别上的正标签的比例的权重向量。

![WPAL.jpg](https://i.loli.net/2020/06/01/fnT6rbLXedS9RDy.jpg)

②AWMT (MM-2017)

众所周知，不同属性的学习难度是不同的。然而，大多数现有算法忽略了这种情况，并在其多任务学习框架中共享相关信息。这将导致负迁移，换句话说，当两个任务不同时，不充分的强力迁移可能会损害学习者的表现。AWMT提出研究一种共享机制，该机制能够动态地、自适应地协调学习不同人属性任务之间的关系。具体来说，他们提出了一个自适应加权的多任务深度框架来联合学习多人属性，并提出了一个验证损失趋势算法来自动更新加权损失层的权重。他们网络的管道可以在下图中找到。

如下图所示，它们采用ResNet-50作为基本网络，并将训练图像和有值图像都作为输入。基本网络将输出训练图像和有值图像的预测属性向量。因此，可以同时获得列车损耗和阀损耗。val损失用于更新权重向量λj(j = 1，...M)，然后利用它来加权不同的属性学习。自适应加权损失函数可表示如下:θ= arg最小值θM X j = 1N X I = 1 <λj，L(ψj(Ii；θ)Lij)>(24)其中，θ表示神经网络的参数，λjis表示对学习jth属性的任务的重要性进行加权的标度值。Ii表示小批量中的第I个图像，Lijis是图像I的属性j的地面真值标签，ψj(Ii；θ)是输入图像Ii在神经网络参数θ下的预测属性。< >是内部产品操作。关键问题是如何自适应地调整权重向量λjin Eq。24.他们提出了验证损失趋势算法来实现这一目标。他们算法背后的直觉是，在同时学习多个任务时，“重要”任务应该被赋予较高的权重(即λj)，以增加相应任务的损失规模。但问题是我们如何知道哪个任务更“重要”，换句话说，我们如何衡量一个任务的重要性？在本文中，作者建议使用泛化能力作为一个客观的衡量标准。具体来说，他们认为一个任务的训练模型具有较低的生成能力，应该设置比其他任务的模型更高的权重。权重向量λjis每k次迭代更新一次，用于计算训练数据的丢失，并在向后传递中更新网络参数θ。他们在多个属性数据集上的实验验证了这种自适应加权机制的有效性。

![AWMT.jpg](https://i.loli.net/2020/06/01/wSeXC7J3Wgv1uZn.jpg)

6.基于课程学习的(Curriculum Learning based Algorithms)

在本趴中，将整理介绍基于课程学习的算法，该算法考虑以“容易”到“困难”的方式学习人类属性，例如:MTCT[1]，CILICIA[2]。

[1] Q. Dong, S. Gong, and X. Zhu, “Multi-task curriculum transfer deep learning of clothing attributes,” in Applications of Computer Vision(WACV), 2017 IEEE Winter Conference on.  IEEE, 2017, pp. 520–529.20, 21
[2] N. Sarafianos, T. Giannakopoulos, C. Nikou, and I. A. Kakadiaris,“Curriculum learning for multi-task classification of visual attributes,” in Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 2608–2615. 20, 21

①MTCT (WACV-2017)

该文提出了一个多任务的课程转移网络来解决手工标注训练数据不足的问题。如下图所示，它们的算法主要包括多任务网络和课程迁移学习。对于多任务网络，他们采用五个堆叠的网络内网络卷积单元和N个并行分支，每个分支代表三层完全连接的子网络，分别用于对N个属性之一建模。模型训练采用软最大损失函数。认知研究表明，人类和动物采用的更好的学习策略是从学习更容易的任务开始，然后逐渐增加任务的难度，而不是盲目地学习随机组织的任务。因此，他们采用课程迁移学习策略进行服装属性建模。

![MTCT.jpg](https://i.loli.net/2020/06/02/vNLc9qzogQI4MJy.jpg)

具体来说，它由两个主要阶段组成。在第一阶段，他们使用干净(即更容易)的源图像及其属性标签来训练模型。在第二阶段，他们嵌入跨域图像对信息，同时将较难的目标图像添加到模型训练过程中，以获取较难的跨域知识。他们采用t-STE (t分布随机三重嵌入)损失函数来训练网络，该网络可以描述为:

$L_{t-STE}=\sum_{I_t},I_{ps},I_{ns}∈T){log\frac{(1+\frac{||f_t(I_t)-f_s(I_{ps})||^2}{\alpha})^\beta}{(1+\frac{||f_t(I_t)-f_s(I_{ps})||^2}{\alpha})^\beta + (1+\frac{||f_t(I_t)-f_s(I_{ns})||^2}{\alpha})^\beta}}$

其中β= 0.5∑(1+α)，α是学生核的自由度。ft(It)和fs(Ips)分别是目标和源多任务网络的特征提取函数

② CILICIA (ICCV-2017) 

与MTCT相似，CILICIA 也将课程学习的思想引入到行人的属性识别任务中，使属性由易到难学习。CILICIA 的管道见下图。他们探索不同属性学习任务之间的相关性，并将这种相关性分为强相关性和弱相关性任务。具体地，在多任务学习的框架下，他们使用各自的皮尔逊相关系数来测量强相关任务，这些强相关任务可以被公式化为:pi = T×j = 1，j6=i cov(yti，ytj) σ(ytiσ(ytj))，i = 1，...其中σ(yti)是任务ti的标签y的标准偏差。前50%的任务与剩余密切相关，可分为强相关组。其余任务属于弱相关组，将在强相关组的知识指导下学习。

对于多任务网络，他们在预测和目标之间采用分类交叉熵函数，其可以定义如下(对于单个属性t):Lt = 1N N X I = 1M X j = 1(1/Mj PM N = 11/Mn)1[yi= j]log(pi，j) (27)，其中如果样本I的目标属于类别j，则1[yi = j]为1，否则为零。属于类j、M和N的样本数分别是类和样本数。为了对不同的属性学习任务进行加权，一个直观的想法是学习另一个用于加权学习的分支网络。然而，作者没有看到这种方法的显著改进。因此，他们采用监督转移学习技术来帮助弱相关组中的属性学习:Lw =λLs+(1λ)Lf w，(28)，其中Lf希望在正向传递过程中使用公式获得总损失。他们还提出了CILICIA -v2 一种有效的方法，以获得任务组使用层次凝聚聚类。它可以是任意数量的，而不仅仅是两组(即强/弱相关)。更具体地说，他们使用计算的皮尔逊相关系数矩阵，使用沃德方差最小化算法来执行分层聚集聚类。Wards方法倾向于生成相同大小的聚类，并分析所有可能的连接聚类对，识别哪个连接产生最小的聚类平方和(WCSS)误差。因此，我们可以通过WCSS阈值操作获得属性组。对于每一组，他们通过仅在聚类内对获得的相应皮尔逊相关系数进行排序来计算聚类的学习序列。一旦所有集群的总依赖关系形成，课程学习过程就可以按降序开始。

![CILICIA.jpg](https://i.loli.net/2020/06/02/bNIXq3WlVu7rwJm.jpg)

小结：

受认知科学最新进展的启发，研究人员还考虑使用这种“容易”到“难”的学习机制来进行PAR。他们将现有的课程学习算法引入到他们的学习过程中，以模拟每个属性之间的关系。诸如自定进度学习的一些其他算法也用于对多标签分类问题或其他计算机视觉任务建模。引入更先进的认知科学著作来指导学习也是值得的。



7.基于图形模型的(Graphic Model based Algorithms)

在许多应用中，图形模型通常用于对结构学习建模。类似地，也有一些工作将这些模型集成到行人属性识别任务中，例如:DCSA [1]，A-AOG [2]，VSGR [3]。

[1] H. Chen, A. Gallagher, and B. Girod, “Describing clothing by semantic attributes,” in Computer Vision – ECCV 2012, A. Fitzgibbon, S. Lazeb-nik, P . Perona, Y . Sato, and C. Schmid, Eds.  Berlin, Heidelberg:Springer Berlin Heidelberg, 2012, pp. 609–623. 1, 4, 22
[2] S. Park, B. X. Nie, and S.-C. Zhu, “Attribute and-or grammar for joint parsing of human pose, parts and attributes,” IEEE transactions on pattern analysis and machine intelligence, vol. 40, no. 7, pp. 1555–1569, 2018. 1, 22, 23
[3] Q. L. X. Z. R. H. K. HUANG, “Visual-semantic graph reasoning forpedestrian attribute recognition,” in Association for the Advancement ofArtificial Intelligence, AAAI, 2019. 1, 22, 23

①DCSA (ECCV-2012)

在该文中，作者建议使用条件随机场来模拟人类属性之间的相关性。

![DCSA.jpg](https://i.loli.net/2020/06/02/bGkJBXZ5zelrfjt.jpg)

如上图示，他们首先使用现成的算法来估计姿态信息，并且仅定位上身的局部部分(下身由于遮挡问题而被忽略)。然后，从这些区域中提取四种基本特征，包括SIFT、 texture descriptor、color in LAB space和skin probabilities。通过SVM融合这些特征来训练多属性分类器。本文的核心思想是应用完全连通的条件随机场来探索属性之间的相互依赖关系。它们将每一个属性函数都视为一个CRF节点，连接每两个属性节点的边反映了这两个属性的联合概率。采用信念传播来优化属性标签成本。

②A-AOG (TP AMI-2018) 

A-AOG模型是属性-与或语法（Attribute and-or grammar）的简称，它明确表示身体各部分的分解和连接，并考虑了姿势和属性之间的相关性。该算法是基于与或图开发的，并且与节点表示分解或依赖关系；or节点代表分解或零件类型的替代选择。

具体来说，它主要整合了三类语法:短语结构语法、依存语法和属性语法。如下图所示。

![attributed parse graph.jpg](https://i.loli.net/2020/06/02/FAtVKjw5WGuXa3h.jpg)

形式上，A-AOG被定义为一个五元组:A-AOG = < S，V，E，X，P > ，其中V是顶点集，它主要包含与-节点，或-节点和终端节点的集合:V = Vand∪Vor∪VT；E是边集，它由两个子集E = Epsg∪ Edg组成:带短语结构语法Epsgand依赖语法Edg的边集。X = {x1，x2，...，xN}是与虚拟环境中的节点相关联的属性集。根据前述定义，解析图可以被公式化为:pg = (V (pg)，E(pg)，X(pg)) 从A-AOG得到的解析图的例子可以在下图中找到。给定图像1，目标是从它们的语法模型中找到最可能的解析图pg。

![A-AOG.jpg](https://i.loli.net/2020/06/02/n2FRraZGpCitN63.jpg)

他们采用贝叶斯框架，该框架将联合后验计算为似然度和先验概率的乘积，以公式表示λ是模型参数。 

他们使用深度神经网络为每个部分生成建议，并采用基于波束搜索的贪婪算法优化上述目标函数。

③VSGR (AAAI-2019) 

该文提出通过视觉语义图推理来估计行人属性。

他们认为，人的属性识别的准确性受到以下因素的严重影响:

1).只有局部零件与某些属性相关；

2).挑战性因素，如姿势变化、视点和遮挡；

3).属性和不同零件区域之间的复杂关系;

因此，他们提出用基于图的推理框架联合建模区域-区域、属性-属性和区域属性的空间和语义关系。在下图中可以找到它们的算法的整体流水线。

![VSGR.jpg](https://i.loli.net/2020/06/02/RcmVKdJIL47TW2M.jpg)

如图所示，该算法主要包含两个子网络，即视觉到语义子网络和语义到视觉子网络。对于第一个模块，它首先将人类图像分成固定数量的局部部分X = (x1，x2，...他们构造了一个图，其节点是局部，边是不同部分的相似性。不同于常规关系建模，它们采用部件之间的相似关系和拓扑结构来连接一个部件和它的相邻区域。相似性邻接矩阵可以表示为:Asa(i，j) = exp(Fs(xi，xj))PMj = 1e exp(Fs(xi，xj)) (35)，其中Fs(Xi，XJ)表示也可以由神经网络建模的每个两部分区域之间的成对相似性。局部部分之间的拓扑关系可以通过以下方法获得:Asl(i，j)= exp(dij/∞)PM j = 1e exp(dij/∞)(36)，其中dijis是两个部分之间的像素距离，而∈是缩放因子。这两个子图通过下面的等式被组合来计算空间图的输出:Gs= AsaXWsa+ AslXWsl ，其中Wsaand和Wslare是两个子图的权重矩阵。

因此，通过卷积后的平均汇集操作可以获得空间上下文表示。在对区域到区域的关系进行编码之后，他们还采用相似的操作来基于空间的上下文对语义属性之间的关系进行建模。新图的节点是属性，它们将属性转换成嵌入矩阵R = (r0，r1，...rK)，其中r0表示“开始”标记，每个列ri是一个嵌入向量。位置编码也被认为利用了属性顺序信息P = (p0，p1，...，pK)。嵌入矩阵和位置编码被组合在一起以获得有序预测路径E = (e0，e1，...，eK)，其中ek= rk+ pk。最后，空间和语义上下文可以通过下式获得:C = E + (Usgs) (38)，其中U是可学习的投影矩阵。对于边，它们只连接第I个节点和下标≤ i的节点，以确保当前属性的预测只与先前已知的输出有关系。连接边的边权重可以通过以下公式计算:Fe(ci，CJ)=φe(ci)TφE0(CJ)(39)，其中φe(87)和φE0(87)是线性变换函数。邻接矩阵也可以通过对每行的连通边权值进行归一化来获得。语义图上的卷积运算可以计算为:ge = aECTwe在语义图上进行卷积后可以得到输出表示，然后用于序列属性预测。语义到视觉的子网络也可以以类似的方式处理，并且它还输出顺序属性预测。这两个子网络的输出被融合作为最终预测，并且可以以端到端的方式被训练。



小结：

由于多种属性之间的关系，人们提出了许多算法来挖掘这些信息。因此，图形模型是第一个思考并被引入到学习管道中的，例如马尔可夫随机场、条件随机场、图神经网络。本小节中审查的工程是通过将图形模型与标准分析相结合的输出。也许其他图形模型也可以用于PAR，以获得更好的识别性能。虽然这些算法有很多优点，但是，这些算法似乎比其他的更复杂。效率问题也需要在实际场景中考虑。



8.其他(Other Algorithms)

本小节用于演示不适用于上述类别的算法，包括:PatchIt [1], FaFS[2], GAM [3].

[1] P . Sudowe and B. Leibe, “Patchit: Self-supervised network weight initialization for fine-grained recognition.” in BMVC, 2016. 1, 24, 26
[2] Y . Lu, A. Kumar, S. Zhai, Y . Cheng, T. Javidi, and R. Feris, “Fully- adaptive feature sharing in multi-task networks with applications in person attribute classification,” in CVPR, vol. 1, no. 2, 2017, p. 6. 1, 24,25
[3] M. Fabbri, S. Calderara, and R. Cucchiara, “Generative adversarial models for people attribute recognition in surveillance,” in Advanced Video and Signal Based Surveillance (A VSS), 2017 14th IEEE International Conference on. IEEE, 2017, pp. 1–6. 1, 24, 25, 26

①PatchIt (BMVC-2016)

常规训练网络通常在辅助任务上采用预训练模型进行权重初始化。然而，它将设计的网络限制为与现有体系结构相似，如AlexNet、VGG或ResNet。与这些算法不同的是，本文提出了一种自监督预训练方法PatchTask，来获得PAR的权值初始化。它的关键见解是利用来自同一个领域的数据作为预训练的目标任务，并且它只依赖于自动生成的而不是人工标注的标签。此外，我们更容易为我们的任务找到大量未标记的数据。对于补丁任务，作者将其定义为一个K类分类问题。如下图所示，他们首先将图像分成多个不重叠的局部面片，然后，让网络预测给定面片的来源。他们使用PatchTask来获得VGG16卷积层的初始化，并应用于PAR。      

 ![PatchIt.jpg](https://i.loli.net/2020/06/02/dbntzSE5gN9fpUy.jpg)

②FaFS (CVPR-2017) 

多任务学习的目标是在这些任务之间共享相关信息，以帮助提高最终的泛化性能。大多数手工设计的深度神经网络都进行共享和特定于任务的特征学习。与现有工作不同，FaFS 被提出来自动设计紧凑的多任务深度学习架构。

该算法从薄的多层网络开始，并在训练过程中以贪婪的方式动态扩展。通过重复上述扩展过程，将创建树状的深层体系结构，并且类似的任务将驻留在同一分支中，直到位于顶层为止。下图（右子图）说明了此过程。下图（左图）给出了瘦网络和VGG-16模型之间的比较。通过同时同步正交匹配追踪（SOMP），通过最小化目标函数来初始化瘦网络的权重参数[125]：A ∗，ω∗（l）= arg minA∈Rd×d0，| w | = d0 | | Wp，l− AWp，lw：|| F，（41）其中Wp，lis是具有d行的第l层的预训练模型的参数。 Wp，l w：表示截尾的权重矩阵仅使行由集合ω索引。此初始化过程是逐层完成的，适用于卷积层和完全连接层。然后，如下图所示，采用分层模型扩展来扩展瘦网络。此操作从输出层开始，并以自顶向下的方式递归地指向较低的层。还应该注意的是，每个分支都与一个任务子集相关联。他们还将相似和不相似的任务根据几对任务之间的相似性概率分为不同的组。

![FaFS.jpg](https://i.loli.net/2020/06/02/DVQE2lge6cSpbAI.jpg)

③GAM (AVSS-2017) 

该文提出使用深度生成模型来处理行人属性的遮挡和分辨率低的问题。

具体而言，它们的整体算法包含三个子网，即属性分类网络，重建网络和超分辨率网络。对于属性分类网络，他们还采用联合的全局和局部部分进行最终属性估计，如下图所示。

他们采用ResNet50提取深度特征，并采用全局平均池获得相应分数。将这些分数融合为最终属性预测分数。为了解决遮挡和低分辨率问题，他们介绍了深层的AM网络的管道。生成对抗网络来生成重构的超分辨率图像。并使用预处理后的图像作为输入到多标签分类网络的属性识别。

![GAM.jpg](https://i.loli.net/2020/06/02/w3cMHneOhvF8rm9.jpg)

## 2.3 发展前景预测 <span id="2.3">

在这一趴，将从个人理解的角度找找行人属性识别的未来改进方向。下面是列出了一些已发布的PAR源代码。

·A summary of the source code

**Algorithm **         Source Code

DeepMAR            https://github.com/dangweili/pedestrian-attribute-recognition-pytorch

Wang *et* *al.*          https://github.com/James-Yip/AttentionImageClass

Zhang *et* *al.*         https://github.com/dangweili/RAP

PatchIt                 https://github.com/psudowe/patchit

PANDA                https://github.com/facebookarchive/pose-aligned-deep-networks

HydraPlus-Net    https://github.com/xh-liu/HydraPlus-Net

WPAL-Net            https://github.com/YangZhou1994/WPAL-network

DIAA                    https://github.com/cvcode18/imbalanced_learning

①More Accurate and Efficient Part Localization Algorithm（更准确高效的零件定位算法）

人可以以一种非常有效的方式识别详细的属性信息，因为我们可以快速浏览特定区域，并根据本地和全局信息来推理属性。因此，设计可以检测局部部分以进行像我们人类一样的准确属性识别的算法是一个直观的想法。

根据之前的part localization algorithm介绍，很容易发现研究人员确实对挖掘人体局部部位更感兴趣。他们使用人工注释或检测到的人体或姿势信息进行零件定位。基于零件的属性识别的总体框架，如下图所示。

![part localization algorithm.jpg](https://i.loli.net/2020/06/02/mgVProKw81QdE2X.jpg)

还有一些算法尝试以弱监督的方式提出统一框架，以共同处理属性识别和定位。认为这也是行人属性识别的良好而有用的研究方向。

② Deep Generative Models for Data Augmentation（用于数据增强的深度生成模型）

近年来，深度生成模型取得了长足的进步，并提出了许多算法，例如：pixel-CNN，pixel-RNN，V AE ，GAN。渐进式GAN和bigGAN等最新作品甚至使人们对这些算法生成的图像感到震惊。

一个直观的研究方向是我们如何使用深度生成模型来处理低质量的人像或数据分配不平衡的问题？已经有很多研究集中在文本，属性或姿势信息的指导下进行图像生成。生成的图像可用于许多其他任务中以进行数据增强，例如，对象检测，人员重新识别，视觉跟踪等。 GAM还尝试生成用于人属性识别的高分辨率图像。设计新算法以根据给定属性生成行人图像以增强训练数据也是值得的。

③Further Explore the Visual Attention Mechanism（进一步探索视觉注意机制）

近年来，引入注意力机制引起了越来越多的研究者的关注。它仍然是当今使用的最受欢迎的技术之一，并且在许多任务中都与各种深度神经网络集成在一起。就像中指出的那样，人类感知的一个重要特性是，人们不会立即处理整个场景。取而代之的是，人类选择性地将注意力集中在视觉空间的各个部分上，以在需要的时间和地点获取信息，并随着时间的流逝结合来自不同注视的信息，以建立场景的内部表示，指导未来的眼球运动和决策。由于感兴趣的对象可以放置在固定的中心，并且固定区域之外的视觉环境（“混乱”）的无关特征自然会被忽略，因此它也大大降低了任务的复杂性。

许多现有的基于注意力的行人属性识别算法都使用可训练的神经网络专注于特征或任务加权。尽管确实提高了整体识别性能，但是，如何准确有效地定位关注区域仍然是一个开放的研究问题。设计新颖的注意力机制或借鉴其他研究领域（如自然语言处理）来进行行人属性识别将是未来的重要研究方向。

④New Designed Loss Functions（新设计的损失函数）

近年来，提出了许多损失函数用于深度神经网络优化，例如（加权）交叉熵损失，对比损失，中心损失，三重损失，焦点失利。 研究人员还为PAR设计了新的损失函数，例如WPAL，AWMT，以进一步改善他们的认知度表现。 这是一个非常重要的研究方向不同损失函数对PAR的影响。

⑤ Explore More Advanced Network Architecture（探索更多高级网络架构）

现有的PAR模型采用大规模数据集（如ImageNet）上现成的预训练网络作为其骨干网络体系结构。

他们很少考虑PAR的独特特性并设计新颖的网络。近年来提出了一些新颖的网络，例如：胶囊网络，外部存储器网络。但是，仍然没有尝试将此类网络用于PAR。也有论文证明，网络架构越深，我们可以获得的识别性能越好。如今，自动机器学习解决方案（AutoML）受到越来越多的关注，并且还发布了许多开发工具来进行开发，例如AutoWEKA，Autosklearn 。

因此，使用上述方法为将来的工作设计用于人属性识别的特定网络将是一个不错的选择。

⑥Prior Knowledge guided Learning（先验知识指导学习）

与常规分类任务不同，行人属性识别由于人类的偏爱或自然约束而总是具有自己的特征。这是挖掘PAR的先验知识或常识的重要研究方向。例如，我们在不同的季节，温度或场合穿着不同的衣服。另一方面，一些研究人员试图利用历史知识（例如：Wikipediak）来帮助改善其整体性能，例如图像标题，物体检测。因此，如何利用这些信息来探索人的属性之间的关系或帮助机器学习模型进一步理解这些属性仍然是一个尚未研究的问题。

⑦Multi-modal Pedestrian Attribute Recognition（多模式行人属性识别）

尽管现有的单模态算法已经在如上所述的一些基准数据集上取得了良好的性能。然而，众所周知，RGB图像对光照、恶劣天气(如:雨、雪、雾)、夜间等都很敏感，我们似乎不可能在全天和全天候实现准确的行人属性识别。但是智能监控的实际需求远远超过这个目标。

我们如何弥合这一差距？一个直观的想法是从其他模态中挖掘有用的信息，例如热传感器或深度传感器，以便与RGB传感器集成。已经有许多工作试图融合这些多模态数据并显著改善它们的最终性能，例如RGB-热跟踪、运动物体检测、人重新识别、RGB-深度物体检测、分割。我们认为多模态融合的思想也有助于提高行人属性识别的鲁棒性。

![RGB and thermal infrared images.jpg](https://i.loli.net/2020/06/02/RhIb4vfWeJ3PrlC.jpg)

如上图所示，这些热图像可以突出人和一些其他穿着或携带的物体的轮廓。

⑧Video based Pedestrian Attribute Recognition（基于视频的行人属性识别）

现有的行人属性识别是基于单个图像的，但在实际场景中，我们经常获取摄像机拍摄的视频序列。虽然在每一个视频帧上运行现有的算法可以是一种直观且简单的策略，但是效率可能是实际应用的瓶颈。

陈等人通过重新注释MAR数据集，提出了基于视频PAR数据集，该数据集最初是为基于视频的人的重新识别而构建的。一般来说，基于图像的属性识别只能利用给定图像的空间信息，由于信息有限，增加了PAR的难度。相反，给定基于视频的PAR，我们可以联合利用空间和时间信息。

好处可以列举如下:

1).我们可以通过定义更动态的人属性，例如“跑步者”，将属性识别扩展到更一般的情况；

2).运动信息可用于推理单个图像中可能难以识别的属性；

3).在视频中学习的一般人物属性可以为其他基于视频的任务提供更有帮助的信息，例如视频字幕、视频对象检测;

因此，如何高效、准确地识别实际视频序列中的人体属性是一个值得研究的问题。

⑨Joint Learning of Attribute and Other Tasks（共同学习属性和其他任务）

将人的属性学习整合到其他与人相关的任务中也是一个有趣而重要的研究方向。

通过将人员属性考虑到相应任务中，已经提出了许多算法，例如：基于属性的行人检测，视觉跟踪，人员重新识别和社会活动分析。

在未来的工作中，如何更好地探索细粒度的人的属性来完成其他任务，以及如何利用其他任务来更好地识别人的属性是一个重要的研究方向。

# 3. 总结 <span id="3">

整理本文，初衷是为研究了解行人属性识别领域有一个比较全面的认知。

首先是介绍有关PAR的背景信息（定义和挑战性因素）、现有基准-包括流行的数据集和评估标准。

之后，我们从两个方面，即多任务学习和多标签学习中，回顾了可用于PAR的算法。然后，我们简要回顾一下PAR算法，首先回顾一些在许多其他任务中已广泛使用的流行神经网络。

然后，我们分析深层的从不同的角度来看，用于PAR的28种算法包括：基于全局，基于部分，基于视觉注意，基于顺序预测，基于新设计的损失函数，基于课程学习，基于图形模型的算法和其他算法。

最后，总结了这份调查论文，并指出了从PAR的9个方面提出的一些可能改进的研究方向。

# 4. 参考列表 <span id="4">

【1】[Pedestrian Attribute Recognition: A Survey](https://sites.google.com/view/ahu-pedestrianattributes/)

【2】[传统学习和深度学习]([https://www.jiqizhixin.com/articles/2019-12-24-14?from=synced&keyword=%E4%BC%A0%E7%BB%9F%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0](https://www.jiqizhixin.com/articles/2019-12-24-14?from=synced&keyword=传统学习和深度学习))

【3】[Benchmark和baseline的区别](https://www.zhihu.com/question/22529709)

【4】[准确率(Accuracy), 精确率(Precision), 召回率(Recall)和F1-Measure](https://blog.argcv.com/articles/1036.c)

【5】[多任务学习概述](https://zhuanlan.zhihu.com/p/27421983)

【6】[多标签学习概述](https://www.cnblogs.com/liaohuiqiang/p/9339996.html)



