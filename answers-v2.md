We thank the reviewers for the detailed reviews and the insightful comments and suggestions 
that could greatly help improve the presentation of our work. 
The replies to the common and distinct questions raised by the reviewers are given as follows.


**Common issues**
1. About the categorization of views. The categorization is primarily based on 
whether the view varies the metadata of the tensor. Next, whether 
the view context modifies the underlying tensor data is the other consideration of the 
categorization since that read operations retrieve the correct values should be ensured. 
According to the analysis of RAR, RAW, WAW, and WAR, the seven view types are enough to 
keep the consistent read and write order with the source statements, and keep the correct 
metadata in the meanwhile.
2. About the experimental setup. The versions of JAX, XLA(TensorFlow), and TorchScript(Pytorch) 
are 0.2.27, 2.7.0 and 1.10.2, respectively. The version of the TVM codebase we use is 0.6. 
The framework we use is Pytorch-like but closed source. The networks
are trained on a node configured with eight Tesla V100 GPUs.
The backbone network, the batch size, and the epoch of Mask-RCNN are Resnet50-fpn, 
2, and 12, respectively. The backbone network, the batch size, and the epoch
of SSD are SSDVGG, 8, and 24, respectively. The optimizer, the learning rate, 
the batch size, and the epoch of Detr are Adam, 1e-4, 2, and 300, respectively.
The dataset of MaskRCNN and SSD are mscoco2017, while the dataset of Detr is 
a closed source. 
3. About the operation fusion types. Our operator fusion mainly contains operations 
which have a perfect loop and are convenient for paralleling, 
including various element-wise operations like activation operations, 
tensor construction operations like arange, and stack, view-related operations,
and some misc operations like where. Most kinds of operations showed up in the 
experimental cases.


**Reviewer1**
1. Sec4.4 enumerates the main schedule primitives we applied. Besides, 
We would like to formalize Sec4.4 to clarify more schedule details.
2. Memory optimization is mainly embodied as the reduction of memory allocations and data movement.
For instance, the profiling of delta2bbox shows that global data movement is reduced %, 
We would like to add a new chart for the quantized description of memory optimizations.
3. Please refer to the **Common issues** for the replies to the view categorization
and experimental setup, respectively.
4. Since computer vision is an important area in the AI community, we choose the typical 
networks in the popular open-source library MMDetection, which contains amounts of 
view-related operations. Besides, our proposal is not limited to MMDetction networks.

**Reviewer2**
1. The source example of Figure 4 illustrates a fabled function, where the tensor C of the six 
statements corresponds to the six tensor view categories labeled in the Tensor C column of the right subgraph 
in Figure 4. Besides, we would like to expand Table 4 to clarify more detailed examples in the python functions.

**Reviewer3**
1. Please refer to the **Common issues** for the replies to the view categorization
and experimental setup, respectively.
2. Delta2bbox is divided into eight device kernels by views and an unsupported operation, 
and the kernels are reduced due to the support of views. Besides, we would like to expand 
sec5.1 to clarify more about how the device kernels of each case in Table 4 are reduced. 
3. Tensor Expression (TE) is used to describe tensor computations in TVM, which is a 
SSA format IR. Hence, non-SSA view syntax in the high-level language could not be 
intuitively expressed in TE, especially in place updates concerning indexing operations.
4. Since TorchScript also provides lots of optimization passes including operation fusion, 
we provide the experimental result as a reference here.

**Reviewer4**
1. Reference semantics allows multiple tensor views in the high-level language to refer 
to the same memory, and modifications in a view would result in modifications on the 
other views of the same tensor. The dependence tracking approach mainly analyzes the 
RAR, RAW, WAW, and WAR data dependence relationships based on the seven categorized tensor views, 
which essentially lies in the reference semantics of various contexts.
2. Please refer to the **Common issues** for the replies to the operation fusion types.