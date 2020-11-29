## 7. Adaptive Risk Minimization: A Meta-Learning Approach for Tackling Group Shift

> tags:
>
> \#Meta Learning
>
> \#Group Shift

### 论文信息

* 文献链接：https://arxiv.org/abs/2007.02931
* 官方博客：https://ai.stanford.edu/blog/adaptive-risk-minimization/
* 2020/11/29，ICLR2021在审？
* 参考资料：量子位的知乎文章[李飞飞点赞「ARM」：一种让模型快速适应数据变化的元学习方法 | 开源](https://zhuanlan.zhihu.com/p/276006514)
* 笔记记于2020/11/29

### 文献总结

提出的方法是**自适应风险最小化adaptive risk minimization（ARM）**，

### 内容
* 背景
  * 这个领域应该叫**unlabeled test time adaptation**，或者说**无标签自适应**，具体的问题叫**Group Shift**
  
    > 举个例子就是新用户（测试）数据的分布与训练数据分布有所**偏移（Group Shift）**，那么未知新用户标签的情况下咋泛化
    
  * **原始的问题**是empirical risk minimization（ERM），显然实际应用的（泛化）对此带来挑战
  * **传统方案**是对测试数据分布加假设，主要的idea是测试的时候拿一批数据，就有了其近似分布，从中获取信息（那岂不是还是偷偷训练...）
  
    > **test time adaptation**, including batch normalization, label shift estimation, rotation prediction, entropy minimization, and more
    
  * 进一步有**motivation**：此类方法的缺点是比较specific，不能general adaptive。要general一点，本文的idea以上面的例子说明，训练数据的信息不一定只靠监督的labels，每个用户都可以有一些meta data（特征）来指导用户数据分组，或者说是调整分布
  
  * 一个指出domain adaption缺点的idea是它们的方法一般在训练过程就可以同时get目标域、源域数据，只能说关注目标域特征，但是和真正的测试泛化有些差距（*我觉得有道理但是说不清楚，那谁能达到测试泛化？*）
  
![哦哈](https://ai.stanford.edu/blog/assets/img/posts/2020-11-05-adaptive-risk-minimization/arm.gif)

![哇哈](https://ai.stanford.edu/blog/assets/img/posts/2020-11-05-adaptive-risk-minimization/methods.png)

方法的一个主要操作是兼容了元学习，方式是采用了类似于元学习中损失（目标）的形式。有两个设计上的idea：

* 第一个假设是训练数据是按组提供的，该假设有好几篇参考文献，看着挺nb——idea来自于要适用于现实世界的问题，以上文的例子好理解

* 第二个假设是观察所有测试点的批次，而不是一次只观察一个点——idea是要用厉害和易于处理的方法

TBU
