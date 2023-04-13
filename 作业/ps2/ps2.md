# PS2

### p1 Logistic Regression: Training stability

(a) 在训练集A上，代码迅速收敛完成训练，仅仅使用30000次迭代。在训练集B上，训练很难完成，迭代了10000000+次参数仍然没有收敛

(b) 

采用绘制图像的方法分析梯度变化：

a数据集：

![image-20230409134456101](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230409134456101.png)

b数据集：

![image-20230409134549129](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230409134549129.png)

- 从这个角度入手我们发现问题：a数据集梯度收敛的很快，而b数据集进入10万轮训练以后梯度就下降的很慢，在15轮后梯度值小的可怜
- 针对这个问题返回`p01_lr.py`可以发现下面的代码：

```python
 learning_rate = 10
```

- 学习率过大导致了梯度在一个极值点附近震荡最终导致了以上结果

----------------



**参考答案：**

**a数据集**

![image-20230409163622720](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230409163622720.png)

**b数据集**

![image-20230409163638307](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230409163638307.png)



- 观察到a集合不是线性可分的，而b集合是完全线性可分的集合
- functional margin：
  - $\hat \gamma^i = y^i(w^Tx^i+b)$
- 由于$y$ = {+1， -1} 而不是{0,1}，所以损失函数
  - $L(\theta) = \frac 1m \sum_{i = 0} ^ mlog(1 + exp(-y^i\theta^Tx^i))$
- 结合线性可分和logistics 回归的性质得到：
  - 对于集合b线性可分的，$y^i\theta^Tx^i>0$恒成立,类似于functional margin 我们可以对$\theta$进行放缩，它可以以缩放，让$J(\theta) \sim 0$无限接近于0，难以达到收敛。
  - 但是对于集合a线性不可分，无法进行缩放，总会存在一个线性边界使得可以使$J(theta)$维持不变

--------------------

（c）

1. 不能，无论怎么调整，依旧满足$J(\theta)\sim 0$的性质
2. 能。学习率逐渐减小，直到小到足以使变化<1e-15即可结束
3. 不能。因为如何进行线性放缩，最终也不会影响训练集本身线性可分的性质
4. 能。这样的话可以加速曲线的收敛并且可以让$J(\theta)$小于或等于0最终维持在一个常数
5. 能。可以使数据集线性不可分。（**答案：But how to control the scale of noise to avoid losing accuracy?**

**(d)**（抄答案的）不容易收到数据集B的影响。

hinge-loss(知乎有）：$J(\theta) = max(0, 1 - \hat yy)= max(0, 1- y\cdot (\theta^Tx +b))$

线性可分的数据集->$\hat y y > 0$, 我们可以对$\theta$ 和b进行缩放，使得$|y| > 1$，最后$J(\theta) = 0$





### p2 Model Calibration(模型校准)

![088b80b83834296c7f1ea67f95f2a93](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/088b80b83834296c7f1ea67f95f2a93.jpg)

![a827a353d1b94583fa72a206a46b684](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/a827a353d1b94583fa72a206a46b684.jpg)

![1f425d07ea979fda231b4579c1d06c0](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/1f425d07ea979fda231b4579c1d06c0.jpg)

### Bayesian Interpretation of Regularization(正则化的贝叶斯解释)

![04a5a817135a7e6624587a71789ca50](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/04a5a817135a7e6624587a71789ca50.jpg)

![341dbde3aaea9057084711feb3d18cd](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/341dbde3aaea9057084711feb3d18cd.jpg)

![867e8a0983742976f440e075ecfbf57](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/867e8a0983742976f440e075ecfbf57.jpg)

![94e4b59f8fadb4a8d3212e6948e3ed9](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/94e4b59f8fadb4a8d3212e6948e3ed9.jpg)

![60e93969f8a46889e7c460d2a86fe62](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/60e93969f8a46889e7c460d2a86fe62.jpg)

![00bfda85a3c4945e44ab0173e833080](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/00bfda85a3c4945e44ab0173e833080.jpg)

### p4 Constructing kernels

![5d17064929b14cc4ceb1578b73ef512](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/5d17064929b14cc4ceb1578b73ef512.jpg)

![e8cc4de04dec7acd44758a6ab2ea95e](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/e8cc4de04dec7acd44758a6ab2ea95e.jpg)

![e27b5b70679f093b2a29e6a6528b39a](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/e27b5b70679f093b2a29e6a6528b39a.jpg)

![baadbb38a801fe85807a07d228704b1](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/baadbb38a801fe85807a07d228704b1.jpg)

![image-20230412142710118](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230412142710118.png)



### p5 Kernelizing the Perceptron

(a)![0eac89fcce3c2bb2deef10549eb0799](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/0eac89fcce3c2bb2deef10549eb0799.jpg)

![19f9b94d6243dc3ba40de92183b292d](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/19f9b94d6243dc3ba40de92183b292d.jpg)



（c)

![image-20230412162145153](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230412162145153.png)

![image-20230412162159029](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230412162159029.png)

- 框架代码中使用了等高线图，我们可以看到dot_kernal的效果很差
- 很明显，数据集是线性不可分的，而dot_kernal实际上对应的原函数$\phi(x) = x$，并没有映射到高维的线性空间，所以效果不好





### p6 Spam classification

(a)![image-20230413204213220](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413204213220.png)

![image-20230413204400366](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413204400366.png)

（b)

![image-20230413204439677](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413204439677.png)

Laplace Smooth

![image-20230413204537650](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413204537650.png)



原题目中提示了需要使用对数，因为在predict的时候，计算出的p实在是太小了，所以通过对数似然的方式，进行计算



最后测试结果：**![image-20230413224859337](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413224859337.png)**



（c)

![image-20230413231245735](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230413231245735.png)
