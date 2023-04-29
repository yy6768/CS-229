# Evaluation Metrics

## 课程内容

- 为什么要使用这些内容？
- 二进制分类器
- 分类器的Metrics
- Class Imbalance问题
- Multi-Class的分类指标



## Motivation

- 我们在训练中通常有一个训练目标（可微函数）
- 理想情况下训练的目标函数就是指标，但是在大部分情况下我们也需要其他的一些指标来衡量我们的模型
- 指标对于展现模型性能非常有效
- 指标也对Debuging model非常有效
- 指标也可以比较不同的模型



## 二分类器

notation

- $X$:输入
- $Y$:二进制输出
- $h(X)$:模型
- 两种模型：
  - 直接输出分类的模型（KNN， Decision tree)
  - 输出实数值的模型：
    - 输出可以是margin（SVM）或者LR，NN（概率）
    - 需要选择阈值





### Scored-based Model

- prevalence = $\frac {p}{p + n}$
- 可以根据score将数据样本排序



## 指标 metrics

### 混淆矩阵

设定的阈值为0.5，形成如下类型的矩阵

![image-20230422084444654](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422084444654.png)

这类矩阵被称为 **混淆矩阵**



### 准确度（Accuracy）

- Acc=(TP + TN) / (TP+ TN + FP + FN)

![image-20230422091534771](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422091534771.png)



### Precision

- Pr = (TP) / (TP + FP)

![image-20230422091846057](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422091846057.png)



### Positive Recall 

- Recall :(TP) / (TP + FN)

![image-20230422091919777](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422091919777.png)

- 也存在 Negative Recall （Spec）：(TN) / (TN + FP)



### F Score

- F1 Score:
  - 算术平均值:$HM = (\frac {\sum\limits_i^mx^i}n)^{-1}$
  - 几何平均$GM = (\prod\limits_i^m x_i)^{\frac 1 n}$

![image-20230422092249640](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422092249640.png)

-  F1使用了算术平均值
- 如果使用集合平均值，则成为G score



上述的指标都是 **点度量**

![image-20230422092727001](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422092727001.png)

- 当你将threshold从0.5改变为0.6，则正负样本预测发生变化，测试的阈值改变被称为 **有效的**

- 反之当从0.6改变为0.61，则改变是无效的

  ![image-20230422092857563](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422092857563.png)

- 阈值下降，precision下降，recall提高，specification下降（负recall)



### 如何权衡这些指标？

#### ROC curve

![image-20230422093323988](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422093323988.png)

- AUC-ROC
- 可以看到一些锯齿，表现出的是有效阈值改变



![image-20230422093812552](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422093812552.png)

- 每当出现一个假正类，曲线就在下降

- 曲线并不会像AUC-ROC一样下降至（0,1）



![image-20230422094434620](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422094434620.png)

- 两个模型拥有相同的正例和反例的排列顺序，所以两者一定拥有AU-ROC和AU-PRC和相同的accuracy

- 使用Log-loss，当模型越有信心认为该样本是正类时，预测错误的惩罚越严重 

![image-20230422094708496](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422094708496.png)

- P(x):predicated Model out
- y：true label

|      | x1   | x2   | x3   |
| ---- | ---- | ---- | ---- |
| P(x) | 0.75 | 0.40 | 1    |
| y    | 1    | 0    | 1    |
| Gain | 0.75 | 0.6  | 1    |



## Class Imbalance

![image-20230422095914568](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422095914568.png)

- prevalence<5%:正例太少
- 指标的值将变得非常不均衡（导致无意义）



#### Imbalance对指标的影响

![image-20230422100133036](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422100133036.png)

- 因为盲目的预测类别为大多数样例拥有的类别，当遇到少数类别时，会预测错误
- Log-loss：Majority class会极度影响log -loss

案例：极度多数为负类，并且按照下图进行排序

![image-20230422100255804](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422100255804.png)

AUROC = $\frac  89 \approx 89%$





## Multi-class 情况下的指标

![image-20230422100638463](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230422100638463.png)

- confusion matrix变为n*n,通常会通过上色 
- 多数metrics （尤其是accuracy）需要乘上一定的比例
- 当类别变得越多，类别不平衡的为题会越来越严重
  - 对于刚才介绍的2进制类别不平衡的问题，多分类情况下也同样适用
- 