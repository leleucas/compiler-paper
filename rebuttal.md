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



##### 实验方面
1）实验设置 （reviewer1【1，3】reviewer3【1】）
2）view对于存储的影响 （reviewer1【1】）
3）对于实验结果的分析 （reviewer3【1】）

##### view的分类方面
1）解释清楚view为什么分成那么几类 （reviewer1【2】reviewer3【3】）
2）reference track分析的方法 （reviewer4【1】）
3）解释清楚类别分别对应什么测例 （reviewer2【1】）

##### schedule优化方面
1）优化细节 （reviewer4【2】reviewer1在comment中也提到了。）

##### 为什么仅选择了mmdection网络 （reviewer1【4】）

##### tvm缺乏哪些view操作的支持 （reviewer3【2】）

##### torchscript不属于compiler。。。（reviewer3【4】）














