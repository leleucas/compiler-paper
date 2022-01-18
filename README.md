##### 文章结构
- introduction 部分介绍jax xla tvm以及mlir对于view支持的不足之处,当前深度学习编译器的优化手段主要是通过算子融合带来的性能提升，上层python虽然因为其易用性被广泛使用，但是部署在高性能加速卡上是一种较难且必须的工作。 （2页）
- backgroud部分介绍端到端的架构简述，添加一个架构图，参考nimple， tvm 中 tensor以及operator的概念， numpy indexing （1页或者1页半）
- 当前还需要添加一个表格，对算法对应的大图，以及DAG遍历使用的小图 
- 补充完善memory optimization，以及schedule primitive （Numpy部分的描述总共4页）
- 实验部分 （2页）
- 相关工作以及conclusion
- 实验部分：elena vs tvm, elena vs 未使用优化的版本，elena vs jax vs torch vs tensorflow, 在不同GPU上的性能对比结果。

##### 近期需求
1. 确定文章结构，确定关键部分的说法
2. 相关工作
- mlir，torch：vtensor部分（许平，可以直接更新在introduction部分以及related work部分）
- tensorIR：（钱超，可以直接更新在introduction部分以及related work部分）
3. 实验部分
- 整网测试
- jax，torchscript，xla： 许平
- elena：丽娟
4. ppt：讲清楚算法流程 丽娟
5. 专利撰写
6. 请教framework的pipeline，请教周洋
7. 时间安排

###### 思路

abstract:
神经网络中除了包含经典的耗时算子convolution外，一些网络层包。。。。，可能非常耗时。如目标检测网络中的后处理阶段往往能占据总网络的30-60%。含大量由pythonic grammitical feature构成的网络层，我们发现view是一种常见的操作，并且在python中，view也是一种基本和重要的优化方法。另一方面，AI算法研究员通常依赖于深度学习框架，深度学习框架耗时算子调用第三方库，而通过深度学习编译器提供在线编译功能，基于Python语言的动态特性，往往只能支持部分的语法特征，并通过算子融合来以及端上的代码生成方式。能够融合越多的算子，性能越高。fit for the ever-changing requirements。 但是对于view的支持却有很多不足，例如XLA， torchscript等。TVM

1.在introduction部分需要说明，一方面用户依赖于深度学习框架提供的api，便捷设计开发新的网络，另一方面，由于python的动态语言特性，编译器往往只能支持部分的python，提供在线编译优化方法。往往算子融合带来主要的性能提升。编译器在python语法方面的表达以及优化对于性能优化带来主要的性能提升。
