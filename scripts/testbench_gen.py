import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import math

def float_to_fixed_point(value, q_bits, i_bits):
    number = value * pow(2,q_bits)
    if (number >= -1.0 and number <= 1.0):
        return bin(0)[2:].zfill(q_bits + i_bits)
    # Determine the sign bit
    sign_bit = '1' if number < 0 else '0'
    number = int(number)

    # Convert the absolute value to binary
    abs_value = abs(value)
    binary_value = bin(abs(number))[2:].zfill(q_bits + i_bits - 1)  # Subtract 1 for the sign bit

    # Perform two's complement if the value is negative
    if value < 0:
        binary_value = twos_complement(binary_value, q_bits + i_bits - 1)

    # Combine the sign bit and the binary value
    binary_representation = sign_bit + binary_value

    return binary_representation


def twos_complement(binary_value, num_bits):
    # Determine the complement value
    complement_value = ''.join('1' if bit == '0' else '0' for bit in binary_value)

    # Add leading zeros if necessary
    if len(complement_value) < num_bits:
        complement_value = '1' + complement_value.zfill(num_bits - 1)

    # Add 1 to the complement value
    twos_complement_value = bin(int(complement_value, 2) + 1)[2:].zfill(num_bits)

    return twos_complement_value

ifm_width = 24
wgt_width = 16
out_width = 24
tile      = 8
tile_out  = 10

# Input image
ih = 227
iw = 227

# Convolution 1
kk = 11
stride = 4
pad = 0
relu = 1
ic = 1
oc = 32

# Max pooling 1
kk_pool = 3
stride_pool = 2

# Convolution 2
kk_1 = 5
stride_1 = 1
pad_1 = 2
relu_1 = 1
oc_1 = 64

# Max pooling 2
kk_pool_1 = 3
stride_pool_1 = 2

# Convolution 3
kk_2 = 3
stride_2 = 1
pad_2 = 1
relu_2 = 1
oc_2 = 128

# Convolution 4
kk_3 = 3
stride_3 = 1
pad_3 = 1
relu_3 = 1
oc_3 = 128

# Convolution 5
kk_4 = 3
stride_4 = 1
pad_4 = 1
relu_4 = 1
oc_4 = 64

# Max pooling 3
kk_pool_2 = 3
stride_pool_2 = 2

# FC1
in_feature_1 = 2304
out_feature_1 = 2048

# FC2
out_feature_2 = 512

# FC3
out_feature_3 = 10

oh_conv = int((ih+2*pad-kk)/stride) + 1
ow_conv = int((iw+2*pad-kk)/stride) + 1
oh_pool = int((oh_conv-kk_pool)/stride_pool) + 1
ow_pool = int((ow_conv-kk_pool)/stride_pool) + 1
oh_conv_1 = int((oh_pool+2*pad_1-kk_1)/stride_1) + 1
ow_conv_1 = int((ow_pool+2*pad_1-kk_1)/stride_1) + 1
oh_pool_1 = int((oh_conv_1-kk_pool_1)/stride_pool_1) + 1
ow_pool_1 = int((ow_conv_1-kk_pool_1)/stride_pool_1) + 1
oh_conv_2 = int((oh_pool_1+2*pad_2-kk_2)/stride_2) + 1
ow_conv_2 = int((ow_pool_1+2*pad_2-kk_2)/stride_2) + 1
oh_conv_3 = int((oh_conv_2+2*pad_3-kk_3)/stride_3) + 1
ow_conv_3 = int((ow_conv_2+2*pad_3-kk_3)/stride_3) + 1
oh_conv_4 = int((oh_conv_3+2*pad_4-kk_4)/stride_4) + 1
ow_conv_4 = int((ow_conv_3+2*pad_4-kk_4)/stride_4) + 1
oh_pool_2 = int((oh_conv_4-kk_pool_2)/stride_pool_2) + 1
ow_pool_2 = int((ow_conv_4-kk_pool_2)/stride_pool_2) + 1
oh = oh_pool_2
ow = ow_pool_2

conv2d = nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=kk, padding=pad, stride = stride, bias=False)
conv2d_1 = nn.Conv2d(in_channels=oc, out_channels=oc_1, kernel_size=kk_1, padding=pad_1, stride = stride_1, bias=False)
conv2d_2 = nn.Conv2d(in_channels=oc_1, out_channels=oc_2, kernel_size=kk_2, padding=pad_2, stride = stride_2, bias=False)
conv2d_3 = nn.Conv2d(in_channels=oc_2, out_channels=oc_3, kernel_size=kk_3, padding=pad_3, stride = stride_3, bias=False)
conv2d_4 = nn.Conv2d(in_channels=oc_3, out_channels=oc_4, kernel_size=kk_4, padding=pad_4, stride = stride_4, bias=False)
fc_1 = nn.Linear(in_features=in_feature_1, out_features=out_feature_1, bias = False)
fc_2 = nn.Linear(in_features=out_feature_1, out_features=out_feature_2, bias = False)
fc_3 = nn.Linear(in_features=out_feature_2, out_features=out_feature_3, bias = False)

## randomize input feature map
#ifm = torch.randint(0, 256, (1, ic, ih, iw))*1.0/255.0 
##ifm = torch.round(ifm)

## randomize weight
#weight = torch.rand(oc, ic, kk, kk)*1.4 - 0.7
##weight = torch.round(weight)
#weight1 = torch.rand(oc_1, oc, kk_1, kk_1)*0.5 - 0.25
##weight1 = torch.round(weight1)
#weight2 = torch.rand(oc_2, oc_1, kk_2, kk_2)*0.3 - 0.15
##weight2 = torch.round(weight2)
#weight3 = torch.rand(oc_3, oc_2, kk_3, kk_3)*0.25 - 0.125
##weight3 = torch.round(weight3)
#weight4 = torch.rand(oc_4, oc_3, kk_4, kk_4)*0.2 - 0.1
##weight4 = torch.round(weight4)
#weight_fc1 = torch.rand(out_feature_1,in_feature_1)*0.07-0.035
##weight_fc1 = torch.round(weight_fc1)
#weight_fc2 = torch.rand(out_feature_2,out_feature_1)*0.08-0.04
##weight_fc2 = torch.round(weight_fc2)
#weight_fc3 = torch.rand(out_feature_3,out_feature_2)*0.16-0.08
##weight_fc3 = torch.round(weight_fc3)

# Transform image for detection
trans = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,), (0.229,))
])

# Load the test set
test_set = datasets.MNIST(root='./data', train=False, transform=trans)
#print(len(test_set))
ifm, label = test_set[torch.randint(1,10000, (1,)).item()]
ifm = ifm.unsqueeze(0)
# Print the label information
#print(f"Label: {label}")
with open("label.txt", "w") as file:
    file.write(str(label))

# Load model parameters
model_alexnet = torch.load('model_98.62.pth', map_location=torch.device('cpu'))

weight = model_alexnet['feature.0.weight']
weight1 = model_alexnet['feature.3.weight']
weight2 = model_alexnet['feature.6.weight']
weight3 = model_alexnet['feature.8.weight']
weight4 = model_alexnet['feature.10.weight']
weight_fc1 = model_alexnet['classifier.0.weight']
weight_fc2 = model_alexnet['classifier.2.weight']
weight_fc3 = model_alexnet['classifier.4.weight']

# setting the weight
conv2d.weight = nn.Parameter(weight)
conv2d_1.weight = nn.Parameter(weight1)
conv2d_2.weight = nn.Parameter(weight2)
conv2d_3.weight = nn.Parameter(weight3)
conv2d_4.weight = nn.Parameter(weight4)
fc_1.weight = nn.Parameter(weight_fc1)
fc_2.weight = nn.Parameter(weight_fc2)
fc_3.weight = nn.Parameter(weight_fc3)

# computing output feature
ofm = conv2d(ifm)
ofm = nn.ReLU()(ofm)
ofm = nn.MaxPool2d(kk_pool, stride = stride_pool)(ofm)
ofm = conv2d_1(ofm)
ofm = nn.ReLU()(ofm)
ofm = nn.MaxPool2d(kk_pool_1, stride = stride_pool_1)(ofm)
ofm_0 = ofm
ofm = conv2d_2(ofm)
ofm = nn.ReLU()(ofm)
ofm = conv2d_3(ofm)
ofm = nn.ReLU()(ofm)
ofm = conv2d_4(ofm)
ofm = nn.ReLU()(ofm)
ofm = nn.MaxPool2d(kk_pool_2, stride = stride_pool_2)(ofm)
ofm_1 = ofm
ofm = torch.flatten(ofm, 1)
ofm = fc_1(ofm)
ofm = nn.ReLU()(ofm)
ofm_fc1 = ofm
ofm = fc_2(ofm)
ofm = nn.ReLU()(ofm)
ofm_fc2 = ofm
ofm = fc_3(ofm)
#ofm = torch.round(ofm)

ifm_np = ifm.data.numpy().astype(float)
#weight_np = weight.data.numpy().astype(float)
#weight1_np = weight1.data.numpy().astype(float)
#weight2_np = weight2.data.numpy().astype(float)
#weight3_np = weight3.data.numpy().astype(float)
#weight4_np = weight4.data.numpy().astype(float)
#weight_fc1_np = weight_fc1.data.numpy().astype(float)
#weight_fc2_np = weight_fc2.data.numpy().astype(float)
#weight_fc3_np = weight_fc3.data.numpy().astype(float)
ofm_np = ofm.data.numpy().astype(float)
#ofm_np_0 = ofm_0.data.numpy().astype(float)
#ofm_np_1 = ofm_1.data.numpy().astype(float)
#ofm_fc1_np = ofm_fc1.data.numpy().astype(float)
#ofm_fc2_np = ofm_fc2.data.numpy().astype(float)

# Reshape the ifm to a 3D array
ifm_3d = ifm_np.reshape(ic, ih, iw)

# Create a ifm.txt file and write the sorted ifm values
with open("ifm.txt", "w") as file:
    for i in range(ic):
        for j in range(ih):
            for k in range(iw):
                s = float_to_fixed_point(ifm_3d[i][j][k], 13, 11) + " "
                file.write(s)
            file.write(f"\n")
        file.write("\n")

# Create a ifm_dec.txt file and write the sorted ifm values
with open("ifm_dec.txt", "w") as file:
    for i in range(ic):
        for j in range(ih):
            for k in range(iw):
                file.write(f"{ifm_3d[i][j][k]} ")
            file.write(f"\n")
        file.write("\n")

## Reshape the weight to a 3D array
#weight_3d = weight_np.reshape(oc, ic, kk, kk)
#weight1_3d = weight1_np.reshape(oc_1, oc, kk_1, kk_1)
#weight2_3d = weight2_np.reshape(oc_2, oc_1, kk_2, kk_2)
#weight3_3d = weight3_np.reshape(oc_3, oc_2, kk_3, kk_3)
#weight4_3d = weight4_np.reshape(oc_4, oc_3, kk_4, kk_4)
#weight_fc1_3d = weight_fc1_np.reshape(out_feature_1, in_feature_1)
#weight_fc2_3d = weight_fc2_np.reshape(out_feature_2, out_feature_1)
#weight_fc3_3d = weight_fc3_np.reshape(out_feature_3, out_feature_2)

## Create a weight file
#with open("weight.txt", "w") as file:
#    for i in range(oc):
#        for j in range(ic):
#            for k1 in range(kk):
#                for k2 in range(kk):
#                    s = float_to_fixed_point(weight_3d[i][j][k1][k2], 12, 1) + " "
#                    file.write(s)
#                file.write(f"\n")
#            file.write(f"\n")
#        file.write("\n")
#
#with open("weight1.txt", "w") as file:
#    for i in range(oc_1):
#        for j in range(oc):
#            for k1 in range(kk_1):
#                for k2 in range(kk_1):
#                    s = float_to_fixed_point(weight1_3d[i][j][k1][k2], 12, 1) + " "
#                    file.write(s)
#                file.write(f"\n")
#            file.write(f"\n")
#        file.write("\n")
#
#with open("weight2.txt", "w") as file:
#    for i in range(oc_2):
#        for j in range(oc_1):
#            for k1 in range(kk_2):
#                for k2 in range(kk_2):
#                    s = float_to_fixed_point(weight2_3d[i][j][k1][k2], 12, 1) + " "
#                    file.write(s)
#                file.write(f"\n")
#            file.write(f"\n")
#        file.write("\n")
#
#with open("weight3.txt", "w") as file:
#    for i in range(oc_3):
#        for j in range(oc_2):
#            for k1 in range(kk_3):
#                for k2 in range(kk_3):
#                    s = float_to_fixed_point(weight3_3d[i][j][k1][k2], 12, 1) + " "
#                    file.write(s)
#                file.write(f"\n")
#            file.write(f"\n")
#        file.write("\n")
#
#with open("weight4.txt", "w") as file:
#    for i in range(oc_4):
#        for j in range(oc_3):
#            for k1 in range(kk_4):
#                for k2 in range(kk_4):
#                    s = float_to_fixed_point(weight4_3d[i][j][k1][k2], 12, 1) + " "
#                    file.write(s)
#                file.write(f"\n")
#            file.write(f"\n")
#        file.write("\n")
#
#with open("weight_fc1.txt", "w") as file:
#    for i in range(out_feature_1 // tile):
#        for j in range(in_feature_1):
#            for k in range(tile):
#                s = float_to_fixed_point(weight_fc1_3d[k+i*tile][j], 12, 1) + " "
#                file.write(s)
#            file.write(f"\n")
#        file.write(f"\n")
#
##with open("weight_fc1_dec.txt", "w") as file:
##    for i in range(out_feature_1 // tile):
##        for j in range(in_feature_1):
##            for k in range(tile):
##                file.write(f"{weight_fc1_3d[k+i*tile][j]} ")
##            file.write(f"\n")
##        file.write(f"\n")
#
#with open("weight_fc2.txt", "w") as file:
#    for i in range(out_feature_2 // tile):
#        for j in range(out_feature_1):
#            for k in range(tile):
#                s = float_to_fixed_point(weight_fc2_3d[k+i*tile][j], 12, 1) + " "
#                file.write(s)
#            file.write(f"\n")
#        file.write(f"\n")
#
#with open("weight_fc3.txt", "w") as file:
#    for i in range(out_feature_3 // tile_out):
#        for j in range(out_feature_2):
#            for k in range(tile_out):
#                s = float_to_fixed_point(weight_fc3_3d[k+i*tile_out][j], 12, 1) + " "
#                file.write(s)
#            file.write(f"\n")
#        file.write(f"\n")
#
##with open("weight_fc3_dec.txt", "w") as file:
##    for i in range(out_feature_3 // tile):
##        for j in range(out_feature_2):
##            for k in range(tile):
##                file.write(f"{weight_fc3_3d[k+i*tile][j]} ")
##            file.write(f"\n")
##        file.write(f"\n")

## Reshape the ifm to a 3D array
#ofm_fc1_3d = ofm_fc1_np.reshape(out_feature_1)
#ofm_fc2_3d = ofm_fc2_np.reshape(out_feature_2)
ofm_3d = ofm_np.reshape(out_feature_3)

# Create a ofm.txt file
with open("ofm.txt", "w") as file:
    for i in range(out_feature_3):
        s = float_to_fixed_point(ofm_3d[i], 12, 10) + " "
        file.write(s)
        file.write("\n")

# Create a ofm_dec.txt file
with open("ofm_dec.txt", "w") as file:
    for i in range(out_feature_3):
        file.write(f"{ofm_3d[i]} ")
        file.write("\n")

## Create a ofm_fc1_dec.txt file
#with open("ofm_fc1_dec.txt", "w") as file:
#    for i in range(out_feature_1):
#        file.write(f"{ofm_fc1_3d[i]} ")
#        file.write("\n")
#
## Create a ofm_fc2_dec.txt file
#with open("ofm_fc2_dec.txt", "w") as file:
#    for i in range(out_feature_2):
#        file.write(f"{ofm_fc2_3d[i]} ")
#        file.write("\n")

## Reshape the ifm to a 3D array
#ofm_3d_0 = ofm_np_0.reshape(oc_1, oh_pool_1, ow_pool_1)
#ofm_3d_1 = ofm_np_1.reshape(oc_4, oh, ow)

## Create a ofm_dec.txt file and write the sorted ifm values
#with open("ofm_in_dec.txt", "w") as file:
#    for i in range(oc_4):
#        for j in range(oh):
#            for k in range(ow):
#                file.write(f"{ofm_3d_1[i][j][k]} ")
#            file.write(f"\n")
#        file.write("\n")

## Create a ofm_dec.txt file and write the sorted ifm values
#with open("ofm_in0_dec.txt", "w") as file:
#    for i in range(oc_1):
#        for j in range(oh_pool_1):
#            for k in range(ow_pool_1):
#                file.write(f"{ofm_3d_0[i][j][k]} ")
#            file.write(f"\n")
#        file.write("\n")
