We thank the reviewers for the detailed reviews and the insightful comments and suggestions. 
The replies to the common and distinct questions raised by the reviewers are given as follows.


**Common issues**
1. About the categorization of views. The categorization is primarily based on whether the view varies the metadata. Next, whether the view context modifies the underlying tensor data is the other consideration since that read operations retrieve the correct values should be ensured. According to the analysis of RAR, RAW, WAW, and WAR, the seven view types are enough to keep the consistent read and write order with the source statements and keep the correct metadata in the meanwhile.
2. About the experimental setup. The versions of JAX, XLA(TensorFlow), TorchScript(Pytorch), and TVM are 0.2.27, 2.7.0, 1.10.2, and 0.6, respectively. The framework we use has not been open source yet. The backbone network, the batch size, the epoch, and the dataset of Mask-RCNN and SSD are (Resnet50-fpn,2,12,mscoco2017) and (SSDVGG,8,24,mscoco2017), respectively. The optimizer, the learning rate, the batch size, the epoch, and the dataset of Detr are (Adam,1e-4,2, 300, a closed source).
3. About the fusion types. The fusion types contain operations that are easy to parallelize, including various element-wise operations, and some misc operations like where. Most operations show up in the experimental cases.


**Reviewer1**
1. Sec4.4 enumerates the main schedule primitives we applied. Besides, we'd like to formalize Sec4.4 to clarify more schedule details.
2.  The profiling of delta2bbox shows that global data movement is reduced %. Besides, we'd like to add a chart for quantized memory optimizations.
3. Please refer to **Common issues** for the replies to the view categorization and experimental setup, respectively.
4. Networks of MMDetection contain amounts of typical view-related operations. However, the proposal is not limited to MMDetction networks.


**Reviewer2**
1. The tensor C of the six statements in the example source function corresponds to six tensor view categories labeled in the Tensor C column of the right subgraph in Figure 4. Besides, we'd like to expand Table 4 to clarify more detailed examples in the python functions.


**Reviewer3**
1. Please refer to **Common issues** for the replies to the view categorization and experimental setup, respectively.
2. Delta2bbox is divided into eight device kernels by views and an unsupported operation, and the kernels are reduced due to the support of views. Besides, we'd like to expand sec5.1 to clarify more details of Table 4. 
3. Non-SSA view syntax could not be intuitively expressed in TE of TVM, especially in-place updates concerning indexing operations.
4. Since TorchScript provides plenty of optimization passes including operation fusion, we provide the result as a reference here.


**Reviewer4**
1. Reference semantics allows that modifications in a view result in modifications on the other views. The dependence tracking approach mainly analyzes the read and write memory activities based on the seven categorized tensor views, which are essentially rooted in the reference semantics.
2. Please refer to **Common issues** for the replies to fusion types.
