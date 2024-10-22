import torch
import torch.nn as nn
import torch.nn.functional as F


# notes based on:
# https://arxiv.org/pdf/1611.07004

# TODO:
# ---1. generator - UNet 
# ---2. discriminator - PatchGAN 70x70 ( returning 30x30 )
# 3. Random jitter 256x256 -> 286x286 -> 256x256 cropping
# ---4. Instance normalization instead of Batch normalization
# 5. Both L1 + cGAN losses


# --- --- --- INFO --- --- ---
# batchsize = 1 for now (they used different with other datasets) (1-10)
# Weights -> Gaussian(0, 0.02)
# Learning rate -> 0.0002
# Adam -> (beta1=0.5, beta2=0.999)

# Ck -> Convolution-InstanceNorm-LeakyReLU (k-filters)
# CDk -> Convolution-InstanceNorm-Dropout-LeakyReLU (50%)
# all (4x4) stride 2 convolutions
# factoring by 2 in upsampling and downsampling

# LeakyReLUs in encoder, ReLUs in decoder

# In U-Net, skip connections between encoder and decoder between i and n-i layers


# --- --- --- GENERATOR --- --- ---
# encoder:
# C64 C128 C256 C512 C512 C512 C512 C512

# decoder:
# CD512 CD512 CD512 C512 C256 C128 C64

# U-Net decoder:
# CD512 CD1024 CD1024 C1024 C1024 C512 C256 C128

# --- --- --- DISCRIMINATOR (70x70) --- --- ---
# C64 C128 C256 C512
# After last layer: convolution to 1-dimensional output, sigmoid activation
# Normalization not applied to the first C64 layer
# Leaky ReLU slope of 0.2


# try to delete norm from bottleneck layer ( Errata )