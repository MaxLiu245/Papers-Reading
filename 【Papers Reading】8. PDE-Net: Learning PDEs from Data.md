# 8. PDE-Net: Learning PDEs from Data

> tags:
>
> \#ODE
>
> \#Convolution
>
> \#Machine Learning

写在前面：最近觉得自己记笔记读文章的方式跟不上现状了，体现在对文章背景等枝枝叶叶的阅读都还行，但是文章的重点常常get不到，希望能总结出更多有用的信息。就从这篇文章开始，写一段小总结

## 论文信息

* 文献链接：
* [Openview]()似乎过了期限就不公开了？
* 2018，发表在
* 参考资料：
* 笔记记于2020/12/03，最后更新于12/03

## 文献总结


<table id="tfhover" class="tftable" border="1">
<tr><th>文献条目</th><th>具体内容</th></tr>
<tr>
<td>Target</td>
<td>
<body>
  <ul type="disc">
    <li>先提一下领域，combination of pdes and neural networks</li>
    <li>本文针对的target是子问题，从pde轨线中拟合其函数</li>
    <li>第三</li>
  </ul>
</body>
</td></tr>
<tr><td>Motivation/idea</td>
<td>
<body>
  <ul type="disc">
    <li>大量ob data可收集👉data driven motivated</li>
    <li>要拟合就上neural network，与之结合</li>
    <li>2 key ideas/objectives: (at same time)<ul><li>accurately predict dynamics of complex systems</li><li>uncover the underlying hidden PDE models</li></ul></li>
  </ul>
</body>
</td>
</tr>
<tr><td>Model</td>
<td>
<body>
  <ul type="disc">
    PDE-Net
    <li>基本结构是deep feed-forward network</li>
    <li>根据上面的2个key ideas，two major components which are jointly trained:
      <ul><li>approximate differential operations by convolutions with properly constrained filters</li>
          <li>approximate the nonlinear response by deep neural networks or other machine learning methods</li></ul>
    </li>
    <li>constrains are carefully designed by fully exploiting the relation between the orders of differential operators and the orders of sum rules of filters</li>
    <li>每一步预测用$\delta t$block，$\delta t$指的是$(7)$式中的步长。具体就是按照文章假设$(7)$式，从轨线网格中通过卷积计算偏导估计，再统一输入网络$F$计算下一步的数值。多个$\delta t$block参数共享合在一起就是PDE-Net，目标是保持长时预测能力，自然的损失就是数据网格上所有点的平方误差和。若限制卷积核则称模型为constrained PDE Net</li>
  </ul>
</body>
</td>
</tr>
<tr><td>Pros and Cons</td>
<td>
<body>
  <ul type="disc">
    Pros:
    <li>相对于常规拟合，more flexible by learning both differential operators and the nonlinear response function of the underlying PDE model</li>
    <li>本文underlying物理规律体现在卷积核上，练出来是啥样就代表什么样的偏导存在</li>
    <li>all filters are properly constrained, which enables us to easily identify the governing PDE models while still maintaining the expressive and predictive power of the network</li>
    <li>constrains are carefully designed</li>
    <li>由于采用多$\delta t$block监督所有数据，可以predict the dynamical behavior for a relatively long time, even in a noisy environment</li>
    Cons:
    <li>data-->data driven</li>
    <li>combine neural network</li>
    <li>第三</li>
    文章自己提的改进方向：
    <li>怎么用在实际数据上（后来的文章应该是通过隐空间做）</li>
    <li>uncover hidden variables which cannot be measured by sensors directly, such as in data assimilation</li>
    <li>learn stable and consistent numerical schemes for a given PDE model based on the architecture of the PDE-Net</li>
  </ul>
</body>
</td>
</tr>
<tr><td>Row:5 Cell:1</td></tr>
<tr><td>Row:6 Cell:1</td></tr>
</table>
<p><small>Created with the <a href="http://www.pcjson.com/htmltable/" target="_blank">HTML Table Generator</a>, edited and modified by Max</small></p>

## 文献细节

* 如何验证learnable filters的有效性？实验比较被限制的卷积核和固定卷积核的结果

* 怎么限制filters的？对moment matrices操作，这个定义只在文章和[外网](https://en.wikipedia.org/wiki/Moment_matrix)上搜到，文章内从$(5)$式开始定义，
mei
* 理论上大的filter可以估计高维偏导，但一方面估计误差加大，另一方面计算更复杂，再者不优美

## 参考文献

TBU
