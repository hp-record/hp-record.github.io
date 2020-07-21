---
title: Merkdown基本语法
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

使用'#' 可以展现1-6级别的标题

```
# 一级标题## 二级标题### 三级标题
```

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



