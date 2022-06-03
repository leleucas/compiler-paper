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
2. About the experimental setup. The versions of JAX, XLA(TensorFlow), TorchScript(Pytorch), 
and TVM codebase we use are 0.2.27, 2.7.0, 1.10.2, and 0.6, respectively. 
The framework we use has not been open source yet.
The backbone network, the batch size, and the epoch of Mask-RCNN and SSD are (Resnet50-fpn, 
2, 12) and (SSDVGG, 8, 24) respectively. The optimizer, the learning rate, 
the batch size, and the epoch of Detr are Adam, 1e-4, 2, and 300, respectively.
The dataset of MaskRCNN and SSD are mscoco2017, while the dataset of Detr is 
a closed source. 
3. About the fusion types. The fusion types mainly contain operations 
which have a perfect loop and are easy for paralleling, 
including various element-wise operations like activation operations and
view-related operations, and some misc operations like where and repeat. 
Most kinds of operations show up in the experimental cases.


**Reviewer1**
1. Sec4.4 enumerates the main schedule primitives we applied. Besides, 
We would like to formalize Sec4.4 to clarify more schedule details.
2. Memory optimization is mainly embodied as the reduction of memory allocations and data movement.
For instance, the profiling of delta2bbox shows that global data movement is reduced %, 
We would like to add a new chart for the quantized description of memory optimizations.
3. Please refer to the **Common issues** for the replies to the view categorization
and experimental setup, respectively.
4. Since computer vision is an important area in the AI community, we choose typical 
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
sec5.1 to clarify more details of Table 4. 
3. Non-SSA view syntax could not be intuitively expressed in TE of TVM, 
especially in place updates concerning indexing operations.
5. Since TorchScript provides plenty optimization passes including operation fusion, 
we provide the result as a reference here.

**Reviewer4**
1. Reference semantics allows multiple views to refer 
to the same memory, and modifications in a view would result in modifications on the 
other views of the same tensor. The dependence tracking approach mainly analyzes the 
RAR, RAW, WAW, and WAR relationships based on the seven categorized tensor views, 
which essentially lies in the reference semantics of various contexts.
2. Please refer to the **Common issues** for the replies to fusion types.
