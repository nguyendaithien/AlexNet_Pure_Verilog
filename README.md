# FRAMEWORK FOR CONVOLUTION NEURAL NETWORKS
Author: 
  **Nguyen Van Luu** - nluu1784@gmail.com
  **Dao Dong Hung** - thanjacodai2@gmail.com
# INTRODUCTION
<span style="font-size: 120 px;">**Overview:**</span>
Convolutional Neural Networks (CNNs) are vital
in artificial intelligence and machine learning, especially for
image processing and recognition. They are widely used in facial
recognition, object detection, and image classification, signifi-
cantly improving system performance and accuracy. However,
deploying CNNs on hardware poses challenges due to their
high computational and memory requirements and the complex
computations arising from the weight-sharing mechanism used
in CNNs. Designing efficient hardware accelerators involves
balancing speed, power consumption, and resource usage.
In this research, the design and implementation of a computation
unit for CNNs include a convolutional accelerator, max-pooling
layer, fully connected layers, and a softmax activation function.
This study utilizes a data flow called weight stationary (WS) to
minimize data movement and reuse partial sums based on spatial
architecture with an array of processing elements. Specifically,
a softmax activation function is implemented using a Look-Up
Table (LUT) technique to construct a complete AlexNet (batch
size N = 1) for the handwritten digit recognition task using
the MNIST dataset and fixed-point representation for data. The
system achieves an accuracy of 98% in software and 95% after
hardware simulation. This system processes the convolutional
layers at a rate of 33.5 frames per second, with DRAM access
per multiply-and-accumulate (MAC) operation being 0.0844 for
the AlexNet model and 0.111 for the VGG-16 model (batch size
N=1), while the total power consumption of the entire network
is 4.87 W.

<span style="font-size: 120 px;">** Key Features**</span>  
(1) A data flow called weight stationary base on spartial
architecture is employed, where weights are kept fixed
within an array of Processing Elements (PEs).

(2) The utilization of hierarchical memory structure and
FIFO asynchronous on-chip buffer reduces the off-chip
memory access and reuse data.

(3) Fixed-point representation to reduce computational com-
plexity and improve hardware efficiency.

(4) Activation function approximation using lookup table
methods: By precomputing and storing the values of the
softmax function in a LUT, we can significantly reduce
the need for complex calculations during inference.


The co-design approach in this project combines both software and hardware to implement a fine-tuned AlexNet model for handwritten digit recognition. On the software side, the model is trained using the MNIST dataset, where a modified AlexNet architecture is used to improve recognition accuracy specific to the task. The model's weights, obtained from training, are converted into a fixed-point representation to be compatible with the hardware requirements.
On the hardware side, the architecture is designed and implemented in RTL (Register Transfer Level), with IP verification performed to ensure the accuracy and reliability of the hardware model. The fixed-point weights from the software are transferred to the hardware environment, where they are integrated into the AlexNet network architecture. This complete hardware network is then used for real-time recognition of handwritten digit images.
The software and hardware components are connected through a feedback loop for image recognition, accuracy calculation, and performance comparison, as illustrated in Fig 1. This collaborative framework enables efficient processing and verification, leveraging both the flexibility of software and the performance of hardware to achieve optimal results.

![Flow_design](https://github.com/user-attachments/assets/bf9d1d0a-eaba-4a79-97c5-1fdf01c50725)
*                         Hình 1: Mô hình CNN sử dụng cho bài toán phân loại ảnh.*


<span style="font-size: 120 px;">**The arrchitecture of CONV layer**</span>
- Figure 2 illustrates the block diagram of the architecture and memory hierarchy of the convolutional accelerator, which includes a PE array, global buffer, controller block and ReLU activation function. This block is responsible for convolution operations, max pooling, ReLU, and fully connected layers. The weights, biases, and input feature maps are stored in off-chip DRAM and are read into the accelerator via buffers to reduce latency when accessing off-chip memory. The memory hierarchy consists of three types: off-chip DRAM, a global buffer (FIFO buffer), and registers within each PE. Each PE in the PE array is responsible for computing a convolution operation or max-pooling and accumulating the result through the internal PE register and a global buffer. The FIFO buffer is closely associated with the PE array in rows.
The accelerator is controlled by finite state machine (FSM) in controller block.

<img src = "https://github.com/user-attachments/assets/a7e5367d-7715-4ac7-9faf-83966dfac30a" alt = "tool" width = "600"/>


<span style="font-size: 120 px;">**Pipeline**</span>
- For example, in cycle n, one IFM is loaded into the process-
ing element (PE) array. In the very next clock cycle, 9 MAC
operations are performed, with each MAC contributing to its
respective partial sum. In the subsequent cycles, these partial
sums are gradually accumulated, with 9 MAC operations
completed per cycle, as shown by the changes in the partial
sums’ states in the figure 4. It can be observed that the partial
sums are progressively completed in the order of the IFM’s
processing sequence. 2-D convolutional operator with pipeline computing: completion level of each partial sum per cycle, the colored cells represent partial sums, where partial sums with the same color share the same completion level.
![1-D conv](https://github.com/user-attachments/assets/f5c941d3-c1d0-4d33-99d3-0668bfecdfde)

<span style="font-size: 150 px;">**Fully Connected layer:**</span>
- In the FC module, there are 8 PE blocks, with each PE block performing a multiplication and accumulation of ifmaps and weights in parallel. The 'psum_in' input of each PE is selected by a multiplexer to distinguish between the weights. To store the output values, I will use 8 registers and a 8-to-1 multiplexer to sequentially capture and output the results. Figure 4 below will illustrate the input and output signals of the datapath of the FC module

![FC_architec](https://github.com/user-attachments/assets/239cef6b-e5cf-46d0-aac9-83b198388653)



<span style="font-size: 150 px;">**Softmax Function:**</span>
- The softmax function is commonly used in the final layer of
a CNN and plays a crucial role in the hardware implementation
of the CNN. The function is given by the formula 3, which
shows that the highest computational cost in the hardware.
![soft_max](https://github.com/user-attachments/assets/58bf6932-571a-4295-8760-15621ff1009e)
-Because of normalization, the input data for the softmax layer in the DNN is generally not too large. In this study’s model, the
input data range is [-5, 5], and the total number of input data
points is 81,920. As described on the table II, the hardware
input data is fixed to 17 bits with 1 sign bit, 3 integer bits,
and 13 fraction bits and the output is 32 bits. The absolute
error of the output and the floating-point result from software
calculations does not exceed 4.5 ×10−6, and the relative error
does not exceed 0.88%.
![softmax](https://github.com/user-attachments/assets/2d255129-2d6e-478e-86f2-cf967aa7fb24)


<span style="font-size: 150 px;">**Verification Environment:**</span>
First, I will conduct testing on the accuracy of the convolution computation IP. Specifically, the environment is described in Figure 5.1, where the inputs to the DUT (Design Under Test) and the weights are randomly generated through a Python script. These inputs will be convolved based on the PyTorch framework, and the output of this process will be used as the reference value for comparison with the DUT's output. The random inputs from the script will be fed into the DUT to obtain the results, which will then be compared with the reference values from the script. This is the testing scenario for the convolution IP
![verifi_model](https://github.com/user-attachments/assets/d76b1c0c-1592-45c4-910b-5bddeb4863c8)
*Fig 5 1: The testing environment for each convolution layer.*


-The hardware deployment of the network, after testing, achieves an accuracy of 95.3%. The execution on FPGA has a slightly lower accuracy (a reduction of 3.3%) compared to the software-based deployment. However, this deployment still ensures that the network can classify handwritten images effectively.
<span style="font-size: 120 px;">**Environment Expiment:**</span>
![Alexnet](https://github.com/user-attachments/assets/217ba784-ff45-45c2-9401-352cfee7cb0e)


![verifi_conv](https://github.com/user-attachments/assets/ac779e13-32f7-4d1d-94d7-45930c6ffcae)

<span style="font-size: 150 px;">**Result:**</span>
- Fig  show the mumber of MAC in ALexNet and VGG-16 model.
![MAC_in_model (1)](https://github.com/user-attachments/assets/70d48d3d-e340-43c6-8b00-2aabae3da2a9)
- Fig 6 show the number of DRAM access in WS dataflow: 
![MAC](https://github.com/user-attachments/assets/a44ffe45-da3b-447a-8497-6d50368d3433)

Finally, the DRAM access speed per MAC is **0.0844**
access/MAC measured on the **AlexNet** network and **0.111**
access/MAC on the **VGG-16** network.

-The design, training, and extraction of post-training param-
eters for the network were carried out on Google Colab with
GPU support (Tesla T4), using the PyTorch library, and all
network weights are of the float data type. The model used
for this experimental task is based on the AlexNet architecture,
which has been fine-tuned to meet the requirements of the task,
as described in Figure 4. The details of the model are shown in
Table 5.3. The model was trained using the Stochastic Gradient
Descent (SGD) method with the following configuration pa-
rameters: Image dataset: MNIST, Number of training samples:
60,000 images, Learning rate: 0.005, Momentum: 0.8, Batch
size: 32, Epochs: 20.
An independent test dataset, separate from the training set,
consisting of 10,000 images containing digits from 0 to 9, is
used for evaluation. The software testing is performed by a
Python-based program. According to Table V, the pre-trained
neural network with float-type weights achieves an accuracy
of 98.6%.
![neural](https://github.com/user-attachments/assets/165a55f2-d095-4999-8aa9-070694cf4086)



![Picture3 (1)](https://github.com/user-attachments/assets/f0adddb5-274a-466e-939c-0b57e6859cd7)
![table_1](https://github.com/user-attachments/assets/c5816ef9-3873-4c34-96e6-5de56207ef12)
![table_2](https://github.com/user-attachments/assets/9e9cff91-f4bc-492f-ab7a-b349146eb656)

<span style="font-size: 150 px;">**Fix-Point Representation**</span>
-To deploy a neural network model onto hardware, all
weights must be converted into fixed-point representations.
To determine the exact number of bits needed for fixed-point
representation, we need to identify the output range of each
layer in the AlexNet network. The weights in the network
layers are trained based on the AlexNet model, which has
been fine-tuned for the handwritten digit recognition task on
the MNIST dataset. This set of weights achieves a recognition
accuracy of 98.62% on the MNIST test set. The figure 7
below visualizes the value range of weights across the layers
of the pre-trained network.
From the figure, it can be observed that most of the weights
in all layers of the network fall within the range of [-1:1].
Therefore, only 1 bit is needed to represent the sign, and no
bits are required for the integer part.
![table_3](https://github.com/user-attachments/assets/441e360e-b0e2-45f7-b7b4-18a0e760c07f)





