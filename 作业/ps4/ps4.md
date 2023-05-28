# CS229 ps 4

## 前言：

- CS229 已经全部完结——包括20讲课程和一些复习课（最后的卷积神经网络没听，听说不是那么好）
- 详细的笔记和作业实现可以参考[yy6768/CS-229: Stanford CS 229 (2018 autumn version） (github.com)](https://github.com/yy6768/CS-229)
  - 前期前12讲的笔记可能都比较草率，具体可以看看后面的
  - 作业都是认真完成的，只是都是手写（字太丑别骂了呜呜）
- 作业四的主题是：
  - 神经网络 - 1
  - 非监督学习 - 3 4 （PCA ICA)
  - 强化学习 - 2 5 6(重要性采样，Bellman收敛，Cartpole的价值迭代算法)
- 部分参考**`maxim5/cs229-2018-autumn.git`**的仓库（第2题和第4题参考较多）

## 1. Neural Networks: MNIST image classification

### Softmax 层求导

- 因为对于每个类softmax函数的输出都与其他类的输入$x_j$有关，在这里要分类讨论

![74716dbfc730aa1592f73f293fb74d2](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/74716dbfc730aa1592f73f293fb74d2.jpg)



### Relu层求导

- 比较简单直接求导即可

  ```python
  # x >0 => part relu(x) = 1 * grad_output else part relu(x) = 0
      grad_outputs[x <= 0] = 0
      return grad_outputs
  ```



### 卷积层求导

参考：[卷积神经网络(CNN)反向传播算法推导 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/61898234)

这里不给出详细推导（有些复杂）：

值得注意的几个点：

- 总体分为对权重的求导和对数据的求导

- 推导过程中有一部利用了卷积的平移性处理了矩阵旋转
- 最后还需要注意，对应代码框架的变量（np.sum一定要弄清楚）

具体代码见仓库



## 2. Off Policy Evaluation And Causal Inference

首先题干有相当多的关键信息：

这道题目希望我们可以对比回归/重要性采样的两种拟合（模拟）策略的方法，如果我们有$\pi_0$基线策略，就可以对我们尚未实现的$\pi_1$进行一个简要的估计

![Regression](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230527135532196.png)

（a)重要性采样：参考[强化学习中的重要性采样(Importance Sampling) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/371156865)

原理的证明相当简单

![c2afff419b1d64ce7c376d43202a0a1](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/c2afff419b1d64ce7c376d43202a0a1.jpg)



（b) 加权重要性采样：

![b52cbc8396e28db5e14e0da7b9fa060](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/b52cbc8396e28db5e14e0da7b9fa060.jpg)

(c)

c问参考了答案

> **Hint**： Consider the case where there is only a single data element in your observational dataset.

我的理解是：考虑单个数据的情况=》有限的采样中是无法拟合整个数据集？

![401c04688cd64e03662b72db8651681](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/401c04688cd64e03662b72db8651681.jpg)

![8d320ced073b8f9d5c28fb9e277a5d0](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/8d320ced073b8f9d5c28fb9e277a5d0.jpg)

(d) 双重鲁棒性检测

查阅资料发现也是一种常见的评估方式：

（i)

![243824f029530ad34ff4e046e90bf2c](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/243824f029530ad34ff4e046e90bf2c.jpg)

这里中间有一部分代数细节略过（E的展开）

(ii) 

![image-20230527141321507](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230527141321507.png)

这两问应该想要表达无论是回归还是重要性采样，双鲁棒都可以正确的无偏估计$\pi_1$



(e)

![image-20230527141412624](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230527141412624.png)

- 如果简单的关系很容易（simple）将其中的关系建模，最后通过$\hat R$估计$\pi_1$
- 但是如果关系很复杂（complicated），那么我们无法通过数值计算去计算期望，只能通过重要性采样（类似蒙特卡洛积分）估计



## 3.PCA

题目不难理解，希望将k维数据压缩到1维的数轴上

![5ff83a1b948d624bfd562d68a358e00](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/5ff83a1b948d624bfd562d68a358e00.jpg)

![5ff83a1b948d624bfd562d68a358e00](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/5ff83a1b948d624bfd562d68a358e00.jpg)



## 4 Independent components analysis

这题实际上就是探讨为什么高斯矩阵不能作为作为s（source）的假设不行，另外引入了拉普拉斯分布对应的ICA

1. 为什么高斯不行

   - 这个sub-question 大量涉及线性代数，我又来参考[矩阵求导公式的数学推导（矩阵求导——进阶篇） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/288541909)，中间的一些代数细节我也给出一部分（行列式求导没给出，比较常用）

     - 梯度为0即为极值点（解析解的方程）

       ![54484258a6e79d84e1f6748d70d405b](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/54484258a6e79d84e1f6748d70d405b.jpg)

     - 代数细节（求导）

     ![image-20230527162609344](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230527162609344.png)

     - 利用正交矩阵的性质说明不是唯一解

       ![3245b001db76bf888737d4b07ce35e2](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/3245b001db76bf888737d4b07ce35e2.jpg)

2. 拉普拉斯

   ![693e0a4b574d3cc8bce9ebe772eb842](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/693e0a4b574d3cc8bce9ebe772eb842.jpg)

3. 代码：

   严格按照我们推理的更新公式：

   *注意：这里需要使用outer（外积函数）*

   ```python
   def update_W(W, x, learning_rate):
       """
       Perform a gradient ascent update on W using data element x and the provided learning rate.
   
       This function should return the updated W.
   
       Use the laplace distribiution in this problem.
   
       Args:
           W: The W matrix for ICA
           x: A single data element
           learning_rate: The learning rate to use
   
       Returns:
           The updated W
       """
   
       # *** START CODE HERE ***
   
       updated_W = W + learning_rate * (np.linalg.inv(W.T) - np.outer(np.sign(W.dot(x)), x.T))
       # *** END CODE HERE ***
   
       return updated_W
   
   
   def unmix(X, W):
       """
       Unmix an X matrix according to W using ICA.
   
       Args:
           X: The data matrix
           W: The W for ICA
   
       Returns:
           A numpy array S containing the split data
       """
   
       # *** START CODE HERE ***
       S = X.dot(W.T)
       # *** END CODE HERE ***
   
       return S
   
   ```



文件只能用AU打开（不知道为什么），听上去是非常清晰的分离了，还是很强的

## 5.Markov decision processes

- 之前推导过：[动态规划算法 (boyuai.com)](https://hrl.boyuai.com/chapter/1/动态规划算法#47-扩展阅读：收敛性证明) 4.7节

实际上就利两个性质：

- $\sum_i a_ib_i< \sum_i b_i\max\limits_a a_i $
- 对于离散概率分布$\sum\limits_i p(i) = 1$

![8da2dde42c2e51304e64eeee45b59e0](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/8da2dde42c2e51304e64eeee45b59e0.jpg)

 ![1b2a4e9099ca9088a2238ab4f674edb](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/1b2a4e9099ca9088a2238ab4f674edb.jpg)





## 6.Reinforcement Learning: The inverted pendulum

- 倒立摆：

  ![](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528112600767.png)

$(x, \dot x, θ,  \dotθ)$

两种动作：左移右移(省略了不操作)

奖励：当前状态是否倒下（倒下-1，伫立着0）



问题解决方案：价值迭代法

- 策略是选择对应价值最大的动作
- 每一次失败时重新估计状态转移函数和奖励累计次数，并以此来计算价值函数$V$(Bellman方程)



详细代码见仓库（难度不大，按照要求实现即可，注意numpy的求和求极值的axis即可）

这里展示部分我认为可能有问题的代码

```python
def update_mdp_transition_probs_reward(mdp_data):
    """
    Update the estimated transition probabilities and reward values in your MDP.

    Make sure you account for the case when a state-action pair has never
    been tried before, or the state has never been visited before. In that
    case, you must not change that component (and thus keep it at the
    initialized uniform distribution).

    Args:
        mdp_data: The data for your MDP. See initialize_mdp_data.

    Returns:
        Nothing

    """

    # *** START CODE HERE ***
    transition_counts = mdp_data['transition_counts']
    num_counts = transition_counts.sum(axis=1)
    num_state, _, action_state = transition_counts.shape
    for i in range(num_state):
        for a in range(action_state):
            if num_counts[i, a] != 0:
                mdp_data['transition_probs'][i, :, a] = transition_counts[i, :, a] / num_counts[i, a] # state->new_state采样的次数除以从state出发的总次数

    reward_counts = mdp_data['reward_counts']
    for k in range(num_state):
        sum_count = reward_counts[k, 1]
        if sum_count != 0:
            # 查看代码注释
            mdp_data['reward'][k] = -reward_counts[k, 0] / sum_count
    # *** END CODE HERE ***

    # This function does not return anything
    return


def update_mdp_value(mdp_data, tolerance, gamma):
    """
    Update the estimated values in your MDP.

    Perform value iteration using the new estimated model for the MDP.
    The convergence criterion should be based on `TOLERANCE` as described
    at the top of the file.

    Return true if it converges within one iteration.

    Args:
        mdp_data: The data for your MDP. See initialize_mdp_data.
        tolerance: The tolerance to use for the convergence criterion.
        gamma: Your discount factor.

    Returns:
        True if the value iteration converged in one iteration

    """

    # *** START CODE HERE ***
    iters = 0
    transition_probs = mdp_data['transition_probs']

    while True:
        iters += 1
        value = mdp_data['value']
        # 价值迭代
        # V'(s) = R(s) + \gamma max(a \in A) \sum(s' \in S) P_{sa}(s')V(s')
        new_value = mdp_data['reward'] + gamma * value.dot(transition_probs).max(axis=1)
        mdp_data['value'] = new_value

        if np.max(np.abs(value - new_value)) < tolerance:  # 上方注释说明如果最大的更改小于tolerance，那么跳出循环
            break

    return iters == 1
    # *** END CODE HERE ***
```

最终结果：



![seed = 0](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528113910665.png)

![seed = 1](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528113922625.png)

![seed = 2](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528113943292.png)

![seed = 3](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528113955385.png)

show_cart函数还能看是怎么运动的，虽然感受一般，可能控制台运行可以比较好：

![show_cart](http://typora-yy.oss-cn-hangzhou.aliyuncs.com/img/image-20230528114114820.png)



## 引用

[卷积神经网络(CNN)反向传播算法推导 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/61898234)

[强化学习中的重要性采样(Importance Sampling) - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/371156865)

[矩阵求导公式的数学推导（矩阵求导——进阶篇） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/288541909)

[maxim5/cs229-2018-autumn: All notes and materials for the CS229: Machine Learning course by Stanford University (github.com)](https://github.com/maxim5/cs229-2018-autumn)
