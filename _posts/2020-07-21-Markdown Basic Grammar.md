---
title: Markdown基本语法
layout: post
categories: Markdown
tags: grammar
excerpt: markdown的基本使用语法
---

# 目录 <span id="home">

* **[1. 引言](#1)**
* **[2. 主要内容](#2)**
* **[3. 总结](#3)**
* **[4. 参考列表](#4)**

# 1. 引言 <span id="1">  

个人博客上传的内容，我现在基本采用markdown格式，但偶尔会查些关于markdown语法使用的一些资料，近期就想着整理下markdown常用的语法规则。

# 2. 主要内容<span id="2">  

# <center><h1>Markdown 基本语法</h1></center>

## **标题**

第一个方法：

​	使用'#' 可以展现1-6级别的标题

```
# 一级标题## 二级标题### 三级标题
```

第二个方法：

​	一级标题ctrl+1

​	二级标题 ctrl+2

​	三级标题ctrl+3

​	四级标题ctrl+4

​	五级标题 ctrl+5

#### 常用快捷键

加粗：Ctrl+B

斜体：Ctrl+I

字体：Ctrl+数字

下划线：Ctrl+U

返回开头：Ctrl+Home

返回结尾：Ctrl+End

生成表格：Ctrl+T

创建链接：Ctrl+K

插入目录：输入[toc]

删除线：两个开头，两个结尾， 一对双波浪线

输入<><> 括号里分别写center /center

<center>居中</center>

表情：happy： :happy:

三个- 分隔线


## **列表**

使用 `*` 或者 `+` 或者 `-` 或者 `1.` `2.` 来表示列表

例如：

```
* 列表1* 列表2* 列表3
```

效果:

- 列表1
- 列表2
- 列表3

## **链接**

使用 `[名字](url)` 表示连接，例如`[Github地址](https://github.com/youngyangyang04/Markdown-Resume-Template)` 

效果：[Github地址](https://github.com/youngyangyang04/Markdown-Resume-Template)

## **添加代码**

* 第一个方法：

**输入三个~即插入代码块，或右键插入代码块。**

* 第二个方法：

对于代码块使用 ` 把代码括起来 例如 `int a = 0;` 或者使用 ``` 把代码块括起来 例如：

\```

var foo = function (bar) { return bar++; };

\```

效果：

```
var foo = function (bar) {  
	return bar++;
	};
	
```

## **添加图片**

添加图片`![名字](图片地址)` 例如`![Minion](https://octodex.github.com/images/minion.png)`

## **html 标签**

Markdown支持部分html，例如这样

```
<center><h1>XXX</h1> </center>
```

## 添加公式

- 点击“段落”—>“公式块”
- 右键插入公式块
- 快捷键Ctrl+Shift+m
- “$$”+回车

以上方式都能打开数学公式的编辑栏。

[常用公式1](https://blog.csdn.net/mingzhuo_126/article/details/82722455) [常用公式2](https://juejin.im/post/5a6721bd518825733201c4a2)

# Markdown 渲染

有如下几种方式渲染Markdown文档。

- 使用github来渲染，也就是把自己的 .md 文件传到github上，就是有可视化的展现，大家会发现github上每个项目都有一个README.md。
- 使用谷歌浏览器安装MarkDown Preview Plus插件，也可以打开markdown文件，但是渲染效果不太好。
- mac下建议使用macdown来打开 markdown文件，然后就可以直接导出pdf来打印了。
- window下可以使用Typora来打开markdown文件，同样也可以直接导出pdf来打印。

# 3. 总结 <span id="3">  

简单的把markdown使用的一些常规知识整理下，但要想熟练使用markdown，无其他捷径，唯手熟尔。

# 4. 参考列表 <span id="4">  

[如何使用markdown来制作一份自己的简历](https://mp.weixin.qq.com/s/33P_Ur7TUeBqrbShuebgYA)



