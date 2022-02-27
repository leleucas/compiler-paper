##### 文章结构
- introduction 部分介绍jax xla tvm以及mlir对于view支持的不足之处,当前深度学习编译器的优化手段主要是通过算子融合带来的性能提升，上层python虽然因为其易用性被广泛使用，但是部署在高性能加速卡上是一种较难且必须的工作。 （2页）
- backgroud部分介绍端到端的架构简述，添加一个架构图，参考nimple， tvm 中 tensor以及operator的概念， numpy indexing （1页或者1页半）
- 当前还需要添加一个表格，对算法对应的大图，以及DAG遍历使用的小图 
- 补充完善memory optimization，以及schedule primitive （Numpy部分的描述总共4页）
- 实验部分 （2页）
- 相关工作以及conclusion
- 实验部分：elena vs tvm, elena vs 未使用优化的版本，elena vs jax vs torch vs tensorflow, 在不同GPU上的性能对比结果。


##### 下次会议需要讨论的内容
- 文章中提到的对于view的解析以及优化方案，相比于SSA是种什么关系，以及对于像mlir中的memref的关系，考虑新增tensorref的概念。【考虑和控制流问题的融合】
- 文章关键部分描述的充实，主要为：在introduction部分新增tensorref的概念讨论，第三章framework或者说算法思想的部分；算法具体实现部分的描述；实验部分文字描述以及实验图2增加一些case的性能测试，扩展成一个双栏图。
- 实验部分的数据问题
- 时间节点讨论
- tensor creation & tensor generation


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

第一段： AI算法包含非卷积层网络，这些网络通常包含pythonic 语法特征，计算并不是非常密集，伴有较多的张量访存操作。并且，伴随着计算密集形的算子，如convolution，通常使用深度手工优化算法，【引用cudnn，cutlass，convolution优化算法，以及其他厂商的优化库】，这些轻量级的网络层加起来在网络训练以及推理阶段的耗时非常大。例如，在目标检测网络中，可以耗时30%-60%。后处理阶段进行region proposal。图1例举了一些典型网络的耗时以及网络结构。给出的函数valid flags，这种代码结构在后处理阶段非常常见。代码。。。行就是view的常见形式。另一方面，这些网络通常基于深度学习框架实现。深度学习框架则依赖于硬件厂商提供的优化算子库为耗时算子提供支持，依赖于深度学习编译器对算子库中没有实现的版本优化。因为，新的AI芯片不断涌入市场，深度学习编译器占据越来越重要的地位。而在框架中的往往绑定了python 语言，编译技术往往需要通过算子融合以及端上的代码生成带来显著的性能提升。深度学习编译器主要focus到张量以及numerical computations on 张量。往往对python的语法特征没有完整的支持。深度学习编译器也需要不断演进以支持不断改进的需求。例如，对于python语法特征的view的支持，view的支持是值得以及充满挑战的。

intermediate representation
我们采用了跟numpy array中一样的存储模型。解释在tvm中tensor的局限性。解释添加metadata来，shape，stride，offset，storage——object来表示不同的view信息，分别解释属性的意义。解释添加属性dde以表示数据依赖关系。

tensor creation
是否需要生成新的张量是根据当前张量是否需要更新metada中的shape，stride，offset信息，以及当前张量中记录的是否是最新的写操作。上述六种张量中只有第2个是不需要新生成张量的，其他都需要创建新的张量。
首先描述所有新增的算子需要更新正确的stride，offset，shape信息【表格中列出对应关系，这里可能需要详细说】，storage address是相同的。
按照图中举例说明新生成张量的过程。分别阐述不同情况下是否需要新生成张量，以及不同情况下生成过程中对于ddt的更新。

数据依赖关系主要包含哪些。ddt的构成。辅助记录最新的写操作或者读操作。例举每种view在生成时的记录以及dde更新方法。 

operation graph
Generally，需要构建operation 有向无环图，并给出operation的拓扑序列，基于该拓扑序列，对operation嵌入优化信息。有向无环图的节点是operation，边是张量，边的入口是生成张量的算子，出口是读取该
张量的operation。在没有view的情况下，可基于operation中存储的输入张量关联关系构建该有向五环图，但是由于view in-place引发的问题，在边构建的时候需要考虑由于WAW以及RAW的关联关系，即记录在dde中
。例如，图所示为源代码构建的有向无环图，图中所有实线表示的边都是基于表达式的RAW关系构建的边，而虚线构建的边则是基于存储在dde关联关系构建的边。


TODO
RAR 关系忘记考虑了，需要在算法描述中更新


###### 存疑
1. 非卷积层网络。non-convolutional layers? light-weight layers. 希望强调非计算密集性的网络层。
2. 除了cudnn，cublas，查阅其他硬件厂商优化的深度学习算子库。
3. introduction第二段view的解释需不需要那么详细
4. 待澄清 numpy array 和 compiler 中的tensor在本文中可互换
5. 仅基于表达式构建的有向五环图，这里的边是由于RAW引起的数据依赖关系。
6. 需不需要DFS算法
7. 待更新算法，考虑RAR数据关联
8. 整网测试，测试多少个网络，比如测试maskrcnn分别测试单线程和8线程，共测试两个网络，还是仅测试单线程，测试4个网络？
9. 现在elena内部用的是津铭的方案，后期数据可能需要换成之前indexing的方案测试……


###### 2022.1.24讨论
1. torchscript，jax部分数据不准确;关掉indexing后，offset2bbox，tblr2bbox正确性有问题；genbaseanchors没有关掉indexing，正确性也有问题。
2. 实验部分：罗列出一个表格，统计出每个case来源于哪个网络，view有哪些操作【参考dlbench】为bechmark取一个名字。
3. 修改下introduction的第一段首先罗列当前深度学习编译器的火热，deep learning framework的整体流程，热点代码调用第三方库，其他元算子调用编译器支持等，然后切入网络结构，引入我们发现的view的例子，可以举出如valid flags这样的实例。
4. introduction第三段对于tvm的描述过于简单，单独对tvm进行描述【揪出view的问题】

###### 2022.2.14
1. 后处理阶段需要有一些术语：bbox以及。。。【周洋】
2. pure function中对于inplace update问题支持的局限性【jax，miya】
3. 可以通过view实现inplace操作，但是inplace与view并不是同一类问题，当前提出的方案也适用于单独的inplace更新问题
4. memory optimization章节的分段讨论，或者分章节讨论【标题更改】
5. 针对python 语法特征的embedding的必要性【tvm或者深度学习编译器】，在加速卡上的代码生成。


通过翻译上层python，转换成domain-specific compiler中间表示而需要新生成的张量，内存空间申请问题是一个值得注意的问题。在torch，mlir等软件的设计上都得到了充分的重视。
