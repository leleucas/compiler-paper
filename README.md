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
- TVM：钱超
- jax，torchscript，xla： 许平
- elena：丽娟
4. ppt：讲清楚算法流程 丽娟
5. 专利撰写
6. 请教framework的pipeline，请教周洋
7. 时间安排
