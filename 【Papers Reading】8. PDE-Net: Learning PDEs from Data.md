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
  </ul>
</body>
</td>
</tr>
<tr><td>Pros and Cons</td>
<td>
<body>
  <ul type="disc">
    Pros:
    <li>more flexible by learning both differential operators and the nonlinear response function of the underlying PDE model</li>
    <li>all filters are properly constrained, which enables us to easily identify the governing PDE models while still maintaining the expressive and predictive power of the network</li>
    <li>constrains are carefully designed</li>
    <li>predict the dynamical behavior for a relatively long time, even in a noisy environment</li>
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

## 参考文献

TBU
