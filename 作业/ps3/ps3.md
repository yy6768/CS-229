# 2018 CS229 ps3

## （1）A Simple Neural Network

![ea63e12bc1a281c9e5a9f84d1377d84](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/ea63e12bc1a281c9e5a9f84d1377d84.jpg)

![8cf0c99b564941a649477da02949631](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/8cf0c99b564941a649477da02949631.jpg)

![b169efb7a69abf3b3ec67d3a56ea15b](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/b169efb7a69abf3b3ec67d3a56ea15b.jpg)



## （2） KL divergence and Maximum Likelihood

- KL散度用于衡量两个概率分布的差异
- $D_{KL}(P||Q) = \sum\limits_{x\in X} P(x) \log \frac{P(x)} {Q(x)}$
  - 其中P(x)>0

![db59938aee2aea60d690abb499cbb97](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/db59938aee2aea60d690abb499cbb97.jpg)

![69f5dabda31689f1bbcee8298ba5f33](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/69f5dabda31689f1bbcee8298ba5f33.jpg)

![36a867c122f589fc5b91360a147f3ee](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/36a867c122f589fc5b91360a147f3ee.jpg)



## （3）KL Divergence, Fisher Information, and the Natural Gradient

![68f30b005b53727a1fffee0b3b7efb7](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/68f30b005b53727a1fffee0b3b7efb7.jpg)

![cb3e034a5ac21281f63867ca9f5a0d8](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/cb3e034a5ac21281f63867ca9f5a0d8.jpg)

![ed88aeda2852193bd7b93ba99b03f08](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/ed88aeda2852193bd7b93ba99b03f08.jpg)

![2886a10a38ac0d43483d24cccc1e71b](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/2886a10a38ac0d43483d24cccc1e71b.jpg)

![1424f613a4bd2c087c45bc768040528](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/1424f613a4bd2c087c45bc768040528.jpg)

![e76faac42ea7d971432c0ebc40e894d](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/e76faac42ea7d971432c0ebc40e894d.jpg)

![5a1515bc370a6395c1b9ee43d37acde](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/5a1515bc370a6395c1b9ee43d37acde.jpg)

![3aecb39f23c9119dc7eef820d8951a5](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/3aecb39f23c9119dc7eef820d8951a5.jpg)

![9bb248362e0d5250d82e9a2dd095f91](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/9bb248362e0d5250d82e9a2dd095f91.jpg)

![e653a7d9b40d39a852382cfdcb06460](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/e653a7d9b40d39a852382cfdcb06460.jpg)

## （4）Semi-supervised EM

![f1c04af7d5692e69afe4992eb4aace1](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/f1c04af7d5692e69afe4992eb4aace1.jpg)

![7ef34fb1aaf2f42f3651ef2006826e2](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/7ef34fb1aaf2f42f3651ef2006826e2.jpg)

![5d4386d4d36d553ce780478f061e246](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/5d4386d4d36d553ce780478f061e246.jpg)

![b23ae60c3e9c8a8f8b5da87bbe13091](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/b23ae60c3e9c8a8f8b5da87bbe13091.jpg)

![9f51381814844aa7d35d186c699cfd9](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/9f51381814844aa7d35d186c699cfd9.jpg)

无监督EM：分布各不相同

![image-20230429201147689](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230429201147689.png)

![image-20230429201059477](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230429201059477.png)

![image-20230429201155861](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230429201155861.png)

semi-EM比较稳定

![image-20230429201214968](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230429201214968.png)

（f)

- Number of iterations taken to converge :
  - unsupervised:[166,120,101]
  - semi:[26,25,24]
- Stability (i.e., how much did assignments change with different random initializations?)
  - semi is stable, unsupervised is unstable
- Overall quality of assignments.
  - semi is better





## （5）

（a) 结果

![image-20230430001814775](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230430001814775.png)



（b)原有大小：一个像素我们需要3*8bit = 24 bit， 现在 只需要 $log_2 16 =4bit$，空间压缩了将近$\frac 1 6$