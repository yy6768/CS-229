# Hoeffding inequality

## Basic probability bounds

- 包括两个基本的不等式
  - Markov不等式和Chebyshev‘s不等式

![2ef5d59461baf7ede722a64bc0e9bee](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/2ef5d59461baf7ede722a64bc0e9bee.jpg)

![2780d2ca521b990d53bdb8612d70413](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/2780d2ca521b990d53bdb8612d70413.jpg)

- Chebyshev's inequality 有一个很重要的结果：方差有限的随机变量，它们的平均值（average)最终会收敛到它们的均值

![image-20230427132754786](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230427132754786.png)

## Moment generating functions

- Moment generating functions （矩母函数）
  - $M_z(\lambda) := \mathbb{E}[exp(\lambda Z)]$
  - 用来强调随机变量Z超过期望的概率有更清楚的指数界

![6ed5307f41838ce44264125ea4f5628](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/6ed5307f41838ce44264125ea4f5628.jpg)

证明：

![912645a7c45cb292bc28b15230c042e](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/912645a7c45cb292bc28b15230c042e.jpg)

- 切诺夫界在累加上的性质很好，多个随机变量相加，也能满足这一性质

![image-20230427140803049](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230427140803049.png)

### 矩母函数

- 矩母函数上界：$\exist C\in R$

![image-20230427145330242](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230427145330242.png)

![image-20230427145412588](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230427145412588.png)

- Rademacher random variable

![image-20230427145626957](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230427145626957.png)



### Hoeffding's Lemma

