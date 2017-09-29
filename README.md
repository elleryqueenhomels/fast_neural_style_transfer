# Fast-Neural-Style-Transfer

Generative Neural Methods Based On Model Iteration

## Description
Using a deep residual convolutional neural network as the image transformation network, and a VGG19 deep convolutional neural network which is pre-trained on ImageNet dataset to define and calculate the perceptual loss functions.<br />For a specific style, we train an image transformation network over Microsoft COCO dataset.<br />After training, we can use the model to transfer the style to any image only needing one forward computation.

This code is based on Johnson et al. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) [2016.03] and Ulyanov et al. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) [2016.09].

## Manual
- The main file `main.py` is a demo, which has already contained training procedure and inferring procedure (inferring means generating stylized images). You can swith these two procedures by changing the flag `IS_TRAINING`.
- By default,
(1) The content images lie in the folder `./images/content/`
(2) The style images lie in the folder `./images/style/`
(3) The weights file of the pre-trained VGG-19 lies in the current working directory. (See `Prerequisites` below)
(4) The MS-COCO images dataset for training lies in the folder `./MS_COCO/` (See `Prerequisites` below)
(5) The checkpoint files of trained models lie in the folder `./models/` (You should create this folder by yourself before training.)
(6) After the inferring procedure, the stylized images will be generated and put in the folder `./outputs/`
- For training, you should make sure (2), (3), (4) and (5) are prepared correctly.
- For inferring, you should make sure (1), (3) and (5) are prepared correctly.
- Of course, you can organize all the files and folders as you want, and what you need to do is just modifying the `main.py` file.

## Prerequisites
- [Pre-trained VGG19 network](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat) (MD5 `8ee3263992981a1d26e73b3ca028a123`) <br/><b>I have provided a convertor in the `tool` folder. It can convert the matlab file into a npz file which is much smaller and easier to process via NumPy.</b> <br/><b>Or simply download my pre-processed</b> [Pre-trained VGG19 network npz format](http://pan.baidu.com/s/1nv4ZQI1) (MD5 `c7ddd13b12e40033b5031ff43e467065`) <b>The npz format is about 80MB while the mat format is about 550MB.</b>
- [Microsoft COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

## Trained Models
I have trained [models](https://pan.baidu.com/s/1i4DLXvZ) over 10 styles: wave, udnie, escher_sphere, flower, scream, denoised_starry, starry_bright, rain_princess, woman_matisse, mosaic. (You can click the "models" to download)

## My Running Environment
<b>Hardware</b>
- CPU: Intel® Core™ i9-7900X (3.30GHz x 10 cores, 20 threads)
- GPU: NVIDIA® Titan Xp (Architecture: Pascal, Frame buffer: 12GB)
- Memory: 32GB DDR4

<b>Operating System</b>
- ubuntu 16.04.03 LTS

<b>Software</b>
- Python 3.6.2
- NumPy 1.13.1
- TensorFlow 1.3.0
- SciPy 0.19.1
- CUDA 8.0.61
- cuDNN 6.0.21

## References
- This project borrowed some ideas and paradigms from Logan Engstrom's [Fast Style Transfer](https://github.com/lengstrom/fast-style-transfer) and Zhiyuan He's [Fast Neural Style Tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow)
- Parts of README formatting were copied from Justin Johnson's [Fast Neural Style](https://github.com/jcjohnson/fast-neural-style) and Zhiyuan He's [Fast Neural Style Tensorflow](https://github.com/hzy46/fast-neural-style-tensorflow)

## Citation
```
  @misc{ye2017fastneuralstyletransfer,
    author = {Wengao Ye},
    title = {Fast Neural Style Transfer},
    year = {2017},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/elleryqueenhomels/fast-neural-style-transfer}}
  }
```

