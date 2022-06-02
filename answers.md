We thank the reviewers for the detailed reviews and the insightful comments and suggestions that could greatly help improve the presentation of our work.
The replies to the specific questions raised by the reviewers are given as follows.


**Reviewer 1:**
1. The schedule strategy in Sec4.4 enumerates the main schedule primitives we applied including a series of loop transformations, operator inlining, 
and thread binding. While other optimizations such as data layout transformation have not been used. We would like to formalize Sec4.4 in the later 
version to clarify more schedule details.
2. The optimizations enabled by the usage of views are in computation performance impovement and memory reuse. The computation performance
impovement lies in operator fusion, which could be quantified with the reduction of device kernels in Table 4 and reduced running time of functions and 
neural networks in Figure 7-9. Memory reuse mainly embodies as the reduction of memory allocations and data movement. We add an experiment to test 
the data movement operations for test cases in sec5.1, e.g the global data movement operations are reduced from * to * for the case delta2bbox. Besides,
by anlyzing the operations of delta2bbox, we find 21 memory allocations could be reduced. We would like to add a new chart for the 
quantized description of memory optimizations in the later version.
3. The categorization is primarily based on whether the view varies the metadata compared with the tensor definition. Next, whether the view context 
modifies the underlying tensor data is the other consideration of the categorization, since that read operations retrieve the correct values
should be ensured. According to the analysis of RAR, RAW, WAW and WAR, the seven view types are enough to keep the consistent read and write order 
with the source satements, and keep the correct metadata in the meanwhile.
4. The JAX, XLA(TensorFlow), and TorchScript(Pytorch) version we use are 0.2.27, 2.7.0 and 1.10.2, respectively. The version of the 
TVM codebase we use is 0.6. The Pytorch-like deep learning framework has not been open source, but could be functionally equal to 
Pytorch 1.10.2. Besides, the tuning mechanism is simple since we aim at element-wise operations. The thread number is set to 64, and the 
thread block number is set to dim/64, where dim is the product of the dimensions of the corresponding output tensor.
5. Since computer vision (CV) is an important area in AI community, we choose the typical or innovative networks 
in the popular open source library MMDetection, which contains amounts of view-related operations and affects the performance. 
It is worth noting that our proposal is not limited to the CV networks. Besides, we would like to explore other networks to 
find the bottleneck operationto enrich the operator fusion.

**Reviewer 2:**
1. The source example of Figure 4 illustrate a fabled function, which include six possible examples corresponding to the categorized tensor views.
The Tensor C column of the right subgraph in Figure 4 labels the six kinds of examples corresponding to each tensor C corresponding to each stage.
Besides, the test cases in our experiment section also include different kinds of example belong to each category. We would like to clarify more
by expanding Table 4 to clarify more example of categorized examples.

**Reviewer 3:**
1. The experiment setup could be referred to the 4th item of the list in our respond to **Reviewer1**.
2. 

**Reviewer 4:**
