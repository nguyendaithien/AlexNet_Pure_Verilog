# AlexNet_Pure_Verilog
This is my Thesis 

<span style="font-size: 120 px;">**Overview:**</span>
This project aims to design a hardware computation unit for a Convolutional Neural Network (CNN) and implement the full AlexNet network for handwritten digit recognition. The computation unit includes a convolutional unit, max pooling, fully connected layers, and ReLU and Softmax activation functions.

<span style="font-size: 120 px;">**Features**</span>  
-Using 3-level hierachy memory to store partial sum temporary data.

-A dataflow called "Weight stationary" based on Processing Element to reuse data.

-Implement sofftmax function by using LUT.

<span style="font-size: 120 px;">**The arrchitecture of system**</span>

![Alexnet](https://github.com/user-attachments/assets/217ba784-ff45-45c2-9401-352cfee7cb0e)

<span style="font-size: 120 px;">**The arrchitecture of CONV layer**</span>

<img src = "https://github.com/user-attachments/assets/a7e5367d-7715-4ac7-9faf-83966dfac30a" alt = "tool" width = "600"/>

![Flow_design](https://github.com/user-attachments/assets/bf9d1d0a-eaba-4a79-97c5-1fdf01c50725)
![FC_architec](https://github.com/user-attachments/assets/239cef6b-e5cf-46d0-aac9-83b198388653)
![1-D conv](https://github.com/user-attachments/assets/f5c941d3-c1d0-4d33-99d3-0668bfecdfde)
![softmax](https://github.com/user-attachments/assets/2d255129-2d6e-478e-86f2-cf967aa7fb24)
![verifi_model](https://github.com/user-attachments/assets/d76b1c0c-1592-45c4-910b-5bddeb4863c8)
![verifi_conv](https://github.com/user-attachments/assets/ac779e13-32f7-4d1d-94d7-45930c6ffcae)
![MAC_in_model (1)](https://github.com/user-attachments/assets/70d48d3d-e340-43c6-8b00-2aabae3da2a9)
![MAC](https://github.com/user-attachments/assets/a44ffe45-da3b-447a-8497-6d50368d3433)
![neural](https://github.com/user-attachments/assets/165a55f2-d095-4999-8aa9-070694cf4086)
![Picture3 (1)](https://github.com/user-attachments/assets/f0adddb5-274a-466e-939c-0b57e6859cd7)
