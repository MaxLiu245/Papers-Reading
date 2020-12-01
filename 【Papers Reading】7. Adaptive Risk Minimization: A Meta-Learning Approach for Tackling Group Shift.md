# 7. Adaptive Risk Minimization: A Meta-Learning Approach for Tackling Group Shift

> tags:
>
> \#Meta Learning
>
> \#Meta Learner
>
> \#Group Shift

## 论文信息

* 文献链接：https://arxiv.org/abs/2007.02931
* 官方博客：[Adapting on the Fly to Test Time Distribution Shift](https://ai.stanford.edu/blog/adaptive-risk-minimization/)
* [Openview](https://openreview.net/forum?id=MA8eT-vUPvZ)，众神博弈🐂
* 2020/11/29，ICLR2021在审？
* 参考资料：量子位的知乎文章[李飞飞点赞「ARM」：一种让模型快速适应数据变化的元学习方法 | 开源](https://zhuanlan.zhihu.com/p/276006514)
* 笔记记于2020/11/29，最后更新于12/01

## 文献总结

提出的方法是**自适应风险最小化adaptive risk minimization（ARM）**，

## 内容
* 背景
  * 这个领域应该叫**unlabeled test time adaptation**，或者说**无标签自适应**，具体的问题叫**Group Shift**
  
    > 举个例子就是新用户（测试）数据的分布与训练数据分布有所**偏移（Group Shift）**，那么未知新用户标签的情况下咋泛化
    
  * **原始的问题**是empirical risk minimization（ERM），显然实际应用的（泛化）对此带来挑战
  * **传统方案**是对测试数据分布加假设，主要的idea是测试的时候拿一批数据，就有了其近似分布，从中获取信息（那岂不是还是偷偷训练...）
  
    > **test time adaptation**, including batch normalization, label shift estimation, rotation prediction, entropy minimization, and more
    
  * 进一步有**motivation**：此类方法的缺点是比较specific，不能general adaptive。要general一点，本文的idea以上面的例子说明，训练数据的信息不一定只靠监督的labels，每个用户都可以有一些meta data（特征）来指导用户数据分组，或者说是调整分布
  
  * 一个指出domain adaption缺点的idea是它们的方法一般在训练过程就可以同时get目标域、源域数据，只能说关注目标域特征，但是和真正的测试泛化有些差距（*我觉得有道理但是说不清楚，那谁能达到测试泛化？*）

* 方法

  * 基本假设：方法的一个主要操作是兼容了元学习，方式是采用了类似于元学习中损失（目标）的形式。有两个设计上的idea：
  
    * 第一个假设是训练数据是按组提供的，该假设有好几篇参考文献，看着挺nb——idea来自于要适用于现实世界的问题，以上文的例子好理解
    
    * 第二个假设是观察所有测试点的批次，而不是一次只观察一个点——idea是要用厉害和易于处理的方法
  
  * 官方博客给出了两张图见下，第一张是优化目标的训练过程，第二张是文章方法的思想来源之一（感觉都是元学习的使用方式）

    ![哦哈](https://ai.stanford.edu/blog/assets/img/posts/2020-11-05-adaptive-risk-minimization/arm.gif)
    ![哇哈](https://ai.stanford.edu/blog/assets/img/posts/2020-11-05-adaptive-risk-minimization/methods.png)

  * 具体的训练目标，我觉得是**一般**的，比较容易理解，还是一般的元学习方法，所以我认为这就是元学习，只是套了一层壳子。**打个比方就是别人写过一篇某领域的综述了，我偏偏要把概念换一下，重新写一遍**。本文虽然说要针对group shift这种问题，解决用户（测试）数据分布偏移的情况，引入元数据进行引导，但方法上没有什么创新，就是一套元学习，当然这是我自己的看法，怼了一波欢迎指正。
  
    另外，本文的方法如果说有什么让人眼前一亮的，那就是优化算法中的Test time adaptation procedure了，在我之前的尝试中都是直接拿练好的base model做预测，meta model就扔了（应该可以在用一下提取元信息的）；本文在测试的时候重复了此过程，从训练数据得到热启动的参数再取调整后的base model，很有MAML+MWN的味道，可以说是**练出来一个元学习器出来**啦！
  
  * 最后放上真正的个人笔记
  
    > 链接：https://pan.baidu.com/s/1yNXaAn1tYwTyfFLaGjQfDg
    
    > 提取码：1234
    
    > 复制这段内容后打开百度网盘手机App，操作更方便哦--来自百度网盘超级会员V4的分享
    
## 补充

> 2020/12/01更新

在[Openview](https://openreview.net/forum?id=MA8eT-vUPvZ)上看了所有相关的讨论，三个审稿的评分分别是3、4、5，在作者一一回答后改为5、6、7。大家普遍的疑问主要是：

* Group shift本身确实比较specific，有人质疑这不就是领域自适应等方向么。的确是，不过本文强调的是具体的group shift，这是第一点；第二点是其中包括了无标签的适应方案，模型不假设在训练时可以从目标域访问任何标记或未标记的示例，还是和一般问题略有区别的

* 本文的ARM作为一种元学习的方法，有人质疑这比较普通，就是一般的元学习。作者回应本文确实是用一般的元学习解决特定问题，但是一般的元学习主要关注标签适应，本文ARM提供了一个框架，扩展元学习工具，具体是无标签的适应，并在group shift上验证了有效性

* 最后还有一点是有人对于模型的insight，或者说对于具体的group shift中分布的shift提出疑问，因为原文确实没有太多提及分布。作者的回答我不太明白，感觉有点含糊，大概是表达实验验证了分布之间的区别，只是粗略解释了元学习适应多个未标记数据点，进而观察分布是合理的

## 参考文献

Zhang, M., Marklund, H., Gupta, A., Levine, S., & Finn, C. (2020). Adaptive Risk Minimization: A Meta-Learning Approach for Tackling Group Shift. ArXiv, abs/2007.02931.
