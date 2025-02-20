---
title: 做工程的一些原则
layout: post
categories: program 
tags: guideline
excerpt: 认识总结的一些工程原则
---

---------

# 目录 <span id="home">

* **[1. 引言](#1)**
* **[2. 主要内容](#2)**
  * **[2.1 代码](#2.1)**
  * **[2.2 源代码管理](#2.2)**
  * **[2.3 开发流程](#2.3)**
  * **[2.4 开发流程](#2.4)**
* **[3. 总结](#3)**
* **[4. 参考列表](#4)**

---------

# 1. 引言 <span id="1">  

------

计算机领域的研究有其鲜明的特点。我们不能说工程就是研究，但在某种意义上却可以说，研究也是工程。它不乏灵光一闪的高光时刻，但更多的还是分析、假设、尝试、验证、改进的艰苦迭代过程。要保证这样一个过程的顺畅，除了专业领域的深厚积累、能打破成规的开放心态，与他人的积极合作以及适当的工具和流程也必不可少。

下文的内容主要针对研究型项目的特点。

# 2. 主要内容<span id="2">  

---

## 2.1 代码 <span id="2.1">

### **了解你的语言和库**

首先我们要熟悉自己使用的编程语言和库，并遵循一定的标准。这不单纯是一种美学上的要求，也是为了工作更有效率，合作更为顺畅。例如：

- 学习新的语言特性

近年来，我们熟悉的编程语言演化都很快。不同的编程语言也有自己的特色。如果不了解这些语言的特性，写出来的代码就会千篇一律，不能充分发挥各种语言的优势。比如，从 C++11 开始引入的一些新特性（auto、lambda 表达式、智能指针、range based for、并发支持……）提供了不少方便。又如，常见的现代编程语言如C++、javascript 和 Python 都或多或少提供了一些函数式编程的支持，善加应用的话可以使代码更为简洁，对开发效率和执行性能都有帮助。再如，Python 中的列表推导和切片，基于 numpy 的 vectorization，灵活又高效。这些都要求我们持续学习，与时俱进，掌握编程语言的新特性。

- 掌握语言提供的库

不要重复造轮子。几乎所有人都熟悉这句话，但这种现象一再发生，因为不是所有人都会花时间去熟悉语言所提供的库。也有些人在编程实践中积累了一些自己的代码库，不舍得抛弃。现在几乎所有的编程语言都提供了常见的容器和相关算法的实现。这些标准库就性能而言不一定最优，但一般可靠性、可移植性具有良好保证，也更易于在与他人的合作中使用。

- 遵循业界的代码规范

代码的风格、规范是一个永远充满争议的话题。在有些人心中甚至可能接近一个信仰问题。业界也有不少广为流传的代码规范。不论你喜欢哪一种，尽可能保持代码风格的稳定，并与你的团队、合作者保持某种程度的一致。

- 善用第三方代码

编程语言所提供的标准库总有可能不敷使用，幸而各种包管理工具（Nuget、npm、pip……）为我们提供了丰富的扩展库。需要的时候，不妨通过这些工具寻找合用的代码。另外，GitHub 上的大量开源项目也是个很好的资源。
与此同时，请务必关注各类第三方库、开源软件的许可协议。有不明白的地方，咨询有经验的人。如果在这个问题上犯错误，可能会导致不好的后果。

### **坚持自己对规范、质量和品味的要求**

研究开始的时候，总是会认真地考虑设计问题，代码实现也相对干净、规范。可是随着项目变得越来越复杂，有大量的参数要调、实验要做，人手也不够，就容易出现“赶工”的情况。最终结果可能是跑出来了，代码却成了一团乱麻，自己也不想再多看一眼，反正论文已经发出去了。然而后面的同学、同事就惨了，要重复验证、要继续改进、要产品化，可是面对的是一个天坑：

- copy/paste 产生的重复代码，出现问题的时候很难改进修正
- 各种魔术数与外部依赖（第三方库和软件、环境变量、注册表、数据文件、存储路径、……） 没有统一的配置，散见于各处
- 各种奇技淫巧，考验阅读者（包括一段时间后的作者本人）的脑洞、理解能力和耐心
- 完全没有注释的复杂算法让人怀疑人生
- 被注释掉而又没有删掉的代码让人无所适从
- ……

种种现象，初看都不是什么大事，但时间长了、积累多了就埋下了隐患。所以要记住：牢记使命，不忘初心。多学习、总结、借鉴好的做法，并一以贯之地践行。

## 2.2 源代码管理 <span id="2.2">

其实世上本没有源代码管理，踩的坑多了，自然就产生了源代码管理的需求。最最基础的需求，一曰备份，二曰版本。一个项目中，除了人以外，最重要的资产可能就是数据和代码了。如果没有源代码管理，那一旦硬盘出问题，或者一招不慎操作失误，导致代码丢失、被覆盖且不可恢复，可能就只能跑路了。

源代码管理是开发流程中的重要一步，那么怎样做源代码管理呢？

在研发过程中，有多人同时参与，不断有新的代码产生，也不断有旧的代码被修改或者删除。如何能让不同的人都能方便地在同一个代码库中工作？如何保证每一次改动都能被记录、可以被回溯？如何确保如果不同人的改动产生冲突，能被检测到并消除？如何支持某个人同时展开多个任务，修正老 bug，实现新功能，在同一个开发环境中做不同的事而又互不冲突？答案是显而易见的：使用现代的主流代码版本控制系统，并选择合适的代码托管服务。微软公司的 Azure DevOps 和 GitHub 就是其中的两个佼佼者。请忘记那些古老的源代码管理方式，去拥抱它们。

工具和环境的使用需要学习，工作的流程需要相应地改进。这个话题太大，水太深，网上的相关资料和讨论也非常多。在此，我们提出最基本的几点：

### **及时提交**

提交（Commit/Checkin）是代码版本控制系统中的基础单位。每有修改，完成一定的功能，一定要及时提交。因为在系统中回滚历史或者签出指定的版本，也是以提交为单位的。如果总是在积累了很多的改变（比如实现了多个功能、修正了若干 bug）之后才提交，那在需要的时候，就很难回到准确的位置。提交本身并不会消耗太多的资源，即使是做一些预防性提交（比如每天下班时），也未尝不可。另外提交越晚，积累的改变越多，代码合并的难度越大。
在提交的时候要写入有意义的附加消息。一个好的消息，让人在查看提交历史的时候，能够快速定位到目标。然而在实践中，我们经常看到一些这样的消息：test、update、change、aaa 等等，这样的消息缺乏价值。

### **正确运用分支**

代码版本控制系统中的另一个重要概念就是分支。每个人实现不同的新功能，做不同的尝试，修正不同的 bug。分支让人能够在主线之外另开战场，等功能开发完成测试通过以后，再合并回主线。某种程度上，我们每一个独立的开发活动，都应该建立独立的分支，完成开发且合并回主线之后就删除该分支。在 GIT 中，分支非常轻量，并不会带来多少额外消耗。Azure DevOps 对分支提供了很好的支持，可以将分支和task、bug 相关联。

### **在远端无 PR 不合并**

前面我们已经提到，在独立分支上完成了开发，就需要将之合并回主线分支。我们不提倡在本地合并回主线分支并推送到远端。事实上，在 Azure DevOps 和 GitHub 中，都可以设置分支策略，禁止对主线分支的直接推送，只能通过 Pull Request 来进行分支的合并。Pull Request 提供了分支合并的统一流程，并可以结合代码复查、持续集成等活动，非常有利于项目的质量控制。

## 2.3 开发流程 <span id="2.3">

对于工业界的产品开发团队，开发流程可能相当复杂，涉及很多个环节，在每一个环节都应用了不同的规范、工具和技术。对于我们常见的小规模研究型项目而言，似乎不必如此大动干戈，不过还是有一些做法值得借鉴。

### **基于 Azure DevOps 或 GitHub 的合作**

Azure DevOps 和 GitHub 并不仅仅是代码托管服务。它们还是很好的开发流程管理工具。Azure DevOps 提供了源代码管理、开发过程 /Work item 管理、Pipeline、测试计划等的详尽支持，并具备很好的用户体验。GitHub 基于 Issue 的项目管理功能颇为简洁，结合第三方插件也能实现不错的体验。

### **结合 Pull Request 的 Code Review**

在多人合作的时候，我们建议对代码的改变进行必要的 Code Review。这一方面是为了找出潜在的错误，提升代码质量；另一方面也是让团队的其他成员了解、熟悉新的代码；对于团队的新人而言，参与 Code Review 还是个不错的学习机会。
Azure DevOps 和 GitHub 都围绕 Pull Request 实现了很好的 Code Review 功能。在创建 Pull Request 的时候，可以指定 Reviewer。参与者可以在网站上以直观的方式查看 Pull Request 并针对某些代码行提出自己的意见，展开讨论。在所有的问题都得到解答之后，参与者可以批准 Pull Request。Pull Request 一旦获得足够的批准，就可以自动合并入目标分支。这个功能极大地改善了 Code Review 的用户体验，并留存了历史记录可以随时方便地回顾。

### **开发阶段的自动测试**

在什么时候写测试代码？测试代码应该占多大的比例？是只测局部的逻辑还是也测试集成的功能？这又是一个见仁见智的问题。对于一次性代码，可能什么测试都意义不大。但如果一段代码会被重复使用，测试就很有必要了。一般而言，测试的目的不外乎是：

- 验证代码的行为和结果符合预期
- 验证对于错误的情况能够正确处理
- 验证代码的性能符合预期

在项目的迭代过程中，很难保证开发人员有足够的精力和兴趣来对自己的新代码进行详尽的手工测试，更不用说对既有功能的重复测试。因此，很有必要将相关的测试转变为代码，由代码来自动完成。一方面这样的自动测试在必要时可以反复进行，另一方面随着项目的进展也可以随时补充。

### **持续集成**

有了自动测试的代码，就可以结合工具更好地进行测试了。测试代码可以随时手动启动，但 Azure DevOps 和 GitHub 提供了更好的方式。以 Azure DevOps 为例，你可以在 Pipeline 中定义 Build Steps，加入必要的测试步骤。可以自定义 Pipeline 的触发方式，比如定时执行、在某个 build 之后执行、在某个 branch 上有集成时执行、在完成 Pull Request 时执行等。Build/ 测试任务在 Azure DevOps 的 Build Agent 上执行，完全不会干扰你的开发工作。
在 Azure DevOps 上执行 Build/ 测试任务还有一个显著的好处：可以发现环境差异带来的 bug。有的时候，开发人员的提交中会无意中遗漏了某些已被改变的代码或数据文件，或者缺少某些依赖项目（第三方库等）。在本机测试中，这些问题很难被发现。而在一个干净的环境中，这些问题都会暴露出来。

## 2.4 不止是代码 <span id="2.4">

前面我们一直都在提代码。但是在一个实际项目中，我们需要关注的远不止是代码。

### **README**

大家应该都见过，在 GitHub 上的很多项目中，根目录下面都有一个 README 文件。在这个文件中，会简要介绍项目的功能，怎样安装，怎样配置开发环境，可能还包括一个简单的上手指南。这样的一个文档，内容不必多详尽，能让新接触的人不经大的挫折就能成功 build/ 部署/运行即可。

### **依赖组件**

我们的软件开发常常会引用第三方的库、软件。像 npm 之类的包管理工具，能将依赖的组件、版本在 json 文件里面明确指定。但是也有很多其他软件不会自动做到这个。比如在 python 中 import 一个库，并不会自动生成依赖描述文件。还有一些其他库和软件，并不能通过现成的包管理机制获得。这就需要我们手工维护依赖描述，并将必要的代码、文件直接加入我们的代码库。这样才能保证参与项目的其他人（或者 Build Agent）能够方便地构建开发测试环境。

### **安装部署**

如果项目的输出目标不是个简单的应用程序或者单纯的算法代码，需要一定的安装部署步骤，切记要保留安装部署的文档或脚本，必要的参数等。

### **模型、数据**

很多项目会用到各种训练出来的模型，或者从各种来源获得的数据。那么，这些模型、数据也需要用合适的方式来维护。必要的时候，也需要纳入版本控制系统，和源代码的版本有一个对应关系。

### **辅助代码**

除了完成主要功能的代码，在一个项目中，往往还有不少辅助性代码，如前面提到过的爬虫代码，数据库初始化脚本，数据清洗程序等。这些代码虽然不被频繁使用，但也是必不可少，应该一起加入代码库。

### **相关资料**

此外，项目涉及的相关资料、文档、参考论文等，最好也集中管理。

# 3. 总结 <span id="3">  

上面说了这么多，大多点到为止，其实每个点后面都有大文章。正如我们一再强调，本文谈及的主要是研究环境下小型项目中一些可供参考的做法。同时也需要牢记：道路千万条，有效第一条。与其等待完美的方案，不如尽早尝试一些可行的做法，并在实践中总结、调整，使之更适合自己。

# 4. 参考列表 <span id="4">  

[有哪些做研究项目时容易被忽视的工程问题？](https://mp.weixin.qq.com/s?__biz=MzI5MDUyMDIxNA==&mid=2247497004&idx=3&sn=91eeb4e4b4ba8ba0ce64188d286ef5b5&chksm=ec1c18d5db6b91c3457c85e2b2e73874dc99dbd0034498c50359d67a53b30902de10f52c529d&mpshare=1&scene=1&srcid=&sharer_sharetime=1591141131366&sharer_shareid=989b0bb833dbad1aaaaf36960593e33d&key=7b0bb172d3b89960265bdc3de28e0ed426a2bf94e6742d07e26ad914ee372ff7e97bb385ffa0cbb75e91c4b21b2131b9e68f39604628730efb588c20836b6e5f864c0b46f34ea29afaaea21dec26b3b5&ascene=1&uin=MTc5MjQxMzEyOQ%3D%3D&devicetype=Windows+10+x64&version=62090070&lang=zh_CN&exportkey=A1uRz5UjN7%2FKpe5qa93Sz4A%3D&pass_ticket=UE04Cv%2BEOVJKXVLGILtbq%2BDGVy9HnHR33uzPMkwnU8lcWwh9hSOig8Ttbu%2FIktXT)



