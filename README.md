# Fast-Neural-Style-Transfer

Generative Neural Methods Based On Model Iteration

## Discription
Using a deep residual convolutional neural network as the image transformation network, and a VGG19 deep convolutional neural network which is pre-trained on ImageNet dataset to define and calculate the perceptual loss functions.<br />For a specific style, we train a image transformation network using Microsoft COCO dataset.<br />After training, we can use it to transfer the style to any image we want.

This code is based on Johnson et al. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) [2016.03] and Ulyanov et al. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) [2016.09].

## Prerequisites
- [Pre-trained VGG19 network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) (MD5 `8ee3263992981a1d26e73b3ca028a123`) <br/><b>I have provided a convertor in the `tool` folder. It can convert the matlab file into a npz file which is much smaller and easier to process via NumPy.</b> <br/><b>Or simply download my pre-processed</b> [Pre-trained VGG19 network npz format](http://pan.baidu.com/s/1nv4ZQI1) (MD5 `c7ddd13b12e40033b5031ff43e467065`) <b>The npz format is about 80MB while the mat format is about 550MB.</b>
- [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

## Requirements
- Python 3.6.0
- NumPy 1.12.1
- TensorFlow 1.0.0
- (Optional) SciPy 0.19.0  if you want to load a matlab file (actually we only use scipy.io.loadmat)
- (Optional) Matplotlib 2.0.0  if you want to show the stylized images directly

## ********************* still underdevelopment *********************
