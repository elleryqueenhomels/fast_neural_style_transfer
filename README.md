# Fast-Neural-Style-Transfer

Generative Neural Methods Based On Model Iteration

Using a deep residual convolutional neural network as the image transformation network, and a VGG19 deep convolutional neural network which is pre-trained on ImageNet dataset to define and calculate the perceptual loss functions.  For a specific style, we train a image transformation network using Microsoft COCO dataset.  After training, we can use it to transfer the style to any image we want.

This code is based on Johnson et al Perceptual Losses for Real-Time Style Transfer and Super-Resolution [2016.03] (https://arxiv.org/abs/1603.08155) and Ulyanov et al Instance Normalization: The Missing Ingredient for Fast Stylization [2016.09] (https://arxiv.org/abs/1607.08022).

Pre-trained VGG network [MD5 8ee3263992981a1d26e73b3ca028a123] (http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)  I provide a convertor in the tool folder. It can convert the matlab file into a npz file which is much smaller and easier to process via NumPy.

# *************** underdevelopment ***************
