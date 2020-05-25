---
title: 行人属性识别的认识
layout: post
categories: PAR
tags: 综述
excerpt: 目前对PAR的认识，随时补充
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

> 引言，是将读者导入论文主题的部分，主要**叙述综述的目的和作用**，概述主题的有关概念和定义，简述所选择主题的历史背景、发展过程、现状、争论焦点、应用价值和实践意义，同时还可限定综述的范围．使读者对综述的主题有一个初步的印象。     

目前，在对行人属性识别进行了解的时候，对它的了解都很片面，没有一个全面宏观的前世今生的认识，于是对行人属性识别相关查到的认识做个总结整理，易便理解。

首先，**什么是行人属性识别（Pedestrian Attribute Recognition, PAR）？**简单来说，行人属性识别就是给定行人图像时挖掘行人目标的属性，即对图像里的行人目标进行检测与提取，识别行人的特征（比如性别、年龄；是否留胡须、戴口罩、戴眼镜；上下衣颜色、戴帽子；是否有携带物等特征） 。

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



2.基于局部的(Part-based Models)

3.基于视觉注意的(Attention-based Models)

4.基于顺序预测的(Sequential Prediction based Models)

5.基于新设计的损失函数的(Loss Function based Models)

6.基于课程学习的(Curriculum Learning based Algorithms)

7.基于图形模型的(Graphic Model based Algorithms)

8.其他(Other Algorithms)



## 2.3 发展前景预测 <span id="2.3">





# 3. 总结 <span id="3">

> 总结部分又称为结论、小结或结语。书写总结时，可以根据主体部分的论述，提出几条**语言简明、含义确切的意见和建议**；也可以对**主体部分的主要内容作出扼要的概括**，并提出作者自己的见解，表明作者赞成什么，反对什么；对于篇幅较小的综述，可以不单独列出总结，仅在主体各部分内容论述完后，用几句话对全文进行高度概括。



# 4. 参考列表 <span id="4">

> 参考列表是综述的原始素材．也是综述的基础，因此，拥有并列出足够的参考文献显得格外重要。它除了表示尊重被引证作者的劳动及表明引用的资料有其科学依据以外，更重要的是为读者深入探讨该主题提供查找有关文献的线索。

【1】[Pedestrian Attribute Recognition: A Survey](https://sites.google.com/view/ahu-pedestrianattributes/)

【2】[传统学习和深度学习]([https://www.jiqizhixin.com/articles/2019-12-24-14?from=synced&keyword=%E4%BC%A0%E7%BB%9F%E5%AD%A6%E4%B9%A0%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0](https://www.jiqizhixin.com/articles/2019-12-24-14?from=synced&keyword=传统学习和深度学习))

【3】[Benchmark和baseline的区别](https://www.zhihu.com/question/22529709)

【4】[准确率(Accuracy), 精确率(Precision), 召回率(Recall)和F1-Measure](https://blog.argcv.com/articles/1036.c)

【5】[多任务学习概述](https://zhuanlan.zhihu.com/p/27421983)

【6】[多标签学习概述](https://www.cnblogs.com/liaohuiqiang/p/9339996.html)

【7】[]()



