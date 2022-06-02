##### reviewer1
- Impact of optimizations: Would it be possible to dissect and differentiate the contributions of each 
of the optimizations enabled by the usage of views? In particular, I'm interested in seeing the impact 
of view on reusing memory and avoiding new memory allocations. 
These results could be presented in a different chart/table.
- The 7 types of views: Are those the only types of views that enable later 
optimizations or is this classification expected to be extended later?
- Please provide that version of the compilers and frameworks use as the baseline. 
Also, please describe the (auto-) tuning mechanism used the define your baseline.
- Why was MMDetection chosen as the main application from which hotspots were extracted?

##### reviewer2
- Could you elaborate the examples of different tensor types.

##### reviewer3
- First of all, please provide more details about the experimental results 
so that they can be reproducible. Even very basic information is not found, 
such as the batch size (or does it evaluate inference workloads with batch-one inputs?). 
Also, it would be interesting to look into more details about the stats of Table 4. 
For example, why does delta2box result in eight kernels previously, and 
how is the proposed compiler able to aggregate them into just two kernels? 
More specific information such as this would help readers understand the true value of the proposal.
- Along the same line of the concern, it is unclear what is exactly missing in TVM. 
The paper repeatedly mentions that some "sophisticated" operations are not supported 
but doesn't seem to actually describe what those sophisticated operations actually mean. 
It is clear from the performance results that some view operations are indeed not supported in TVM, 
however, as mentioned above, the experimental section is missing a lot of technical details.
- Regarding the categorization of views, it would be helpful to have more discussions and insights on why that categorization is chosen.
- Minor comments: TorchScript itself is not a compiler, 
so comparing it with DL compilers such as XLA and JAX seems odd (e.g., the last paragraph in page 2).

##### reviewer4
- The paper mentions reference semantics a few times. However, the exact analysis used to track dependencies 
between readers and writers to a tensor using their views is not clearly explained. An you explain your dependence tracking approach?
- Loop fusion is mentioned a few times. Typically neural networks are structured as graphs and allow numerous choices in what operations 
are fused and to what extent. The state of the art frameworks considered go to great lengths to explore this space. Can you clarify what 
types of fusion are performed in this work, and which ones show up in the benchmarks considered?


##### view的分类方面
- 解释清楚view为什么分成那么几类 （reviewer1【2】reviewer3【3】）
- reference track分析的方法 （reviewer4【1】）
- 解释清楚类别分别对应什么测例 （reviewer2【1】）

##### schedule优化方面
- 优化细节 （reviewer4【2】reviewer1在comment中也提到了。）
- view对于存储的优化，以及跟operator优化的关系。

##### 实验方面
- 实验设置 （reviewer1【1，3】reviewer3【1】）
- tvm缺乏哪些view操作的支持 （reviewer3【2】）
- view对于存储的影响 （reviewer1【1】）
- 对于实验结果的分析 （reviewer3【1】）

##### 为什么仅选择了mmdection网络 （reviewer1【4】）
##### torchscript不属于compiler。。。（reviewer3【4】）


1. Elaborations on view categories and dependence tracking. 
- The categorization is primarily based on whether the view varies the metadata compared with the tensor definition. Next, whether the view context 
modifies the underlying tensor data is the other consideration of the categorization. According to the RAR, RAW, WAW and WAR, different view contexts are exploited to keep the consistent read and write order with the source satements. 【读写关系分析的关键在于保证每个operation都可以读到正确的值，分类是为了后续依赖关系，或者说每个读操作追踪到了距离该语句最后一次写操作。】Besides, we believe seven categories are enough for the isssue discussed in the aricle.
- Reference semantic allows multiple tensor views in the high-level language refer to the same memory, and modifications in a view would result in modifications on the other views of the same tensor. The dependence tracking approach mainly analyzes the RAR, RAW, WAW and WAR data dependence relationships based the seven categorized tensor views, which essentially lies in the reference semantic of various contexts.
- The source example in Figure 4 enumerates six types of tensor views. Taking tensor C as illustration, tensor C in stage 1 refers to TWD since the corresponding operation define the tensor. Tensor C in stage 2 refers to TRD since the operation reads the definition data. Tensor C in stage 3 refers
to VRD since the operation reads the definition data and varies metadata of C. Tensor C in stage 4 refers to VWS since the operation modifies data after the definition and varies metadata of C. Tensor C in stage 5 and stage 6 are VRS since the operation reads the modified data, and the operation in stage 6
varies the metadata of C. If we use the statement C = E + k to substitute the operation in stage 4, then the substituted C refers to TWS since the operation modifies the definition data and has not changed the metadata.

2. Optimizations.
- In this article our fusion scope includes 116 basic operations including various arithmetic and logical operations, tensor construction operations like stack and permutation, advanced indexing operations, activation operations, view-related operations and some misc operations like where. Besides, most arithmetic and logical operations, tensor construction operations and view-related operations show up in the experimental cases we use.
- The proposal of views in the article brings optimizations both on performance and memory utilization. In terms of performance, the support of views 
enlarge the fusion scope of operators, which has shown in the experimental section. In terms of memory utilization, the proposal reduces memory allocations. For instance, it reduces 21 memory allocations for the Python function delta2bbox. We would like to build a new chart to show the memory
optimization improvment in the later versions of the article. 【或者通过profiler看下局部或者全局访存操作】

3. Experiment technique details.
- Pytorch version， Jax version，Tensorflow version，TVM version【*】？整网训练时候的基本设置【liujun】。tuning的基本设置。
- TVM不支持哪些操作。【*】
- 为什么我们只选用了mmdet网络。【*】
- 为什么我们比较了torchscript
- 对于Table4的细致分析。














