# Fast-Neural-Style-Transfer

Generative Neural Methods Based On Model Iteration

## Description
Using a deep residual convolutional neural network as the image transformation network, and a VGG19 deep convolutional neural network which is pre-trained on ImageNet dataset to define and calculate the perceptual loss functions.<br />For a specific style, we train an image transformation network over Microsoft COCO dataset.<br />After training, we can use the model to transfer the style to any image only needing one forward computation.

This code is based on Johnson et al. [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) [2016.03] and Ulyanov et al. [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022) [2016.09].

## Results
| style | output (generated image) |
| :----: | :----: |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/wave_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/wave-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/udnie-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/escher_sphere-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/flower_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/flower-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/scream_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/scream-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/denoised_starry_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/denoised_starry-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/starry_bright_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/starry_bright-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/rain_princess_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/rain_princess-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/woman_matisse_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/woman_matisse-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/mosaic-Lance.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/wave_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/wave-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/udnie-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/escher_sphere-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/flower_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/flower-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/scream_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/scream-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/denoised_starry_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/denoised_starry-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/starry_bright_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/starry_bright-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/rain_princess_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/rain_princess-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/woman_matisse_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/woman_matisse-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/mosaic-stata.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/wave_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/wave-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/udnie_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/udnie-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/escher_sphere_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/escher_sphere-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/flower_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/flower-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/scream_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/scream-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/denoised_starry_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/denoised_starry-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/starry_bright_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/starry_bright-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/rain_princess_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/rain_princess-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/woman_matisse_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/woman_matisse-brad_pitt.jpg)  |
|![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/images/style_thumb/mosaic_thumb.jpg)|  ![](https://github.com/elleryqueenhomels/fast_neural_style_transfer/blob/master/outputs/mosaic-brad_pitt.jpg)  |

## Manual
- The main file `main.py` is a demo, which has already contained training procedure and inferring procedure (inferring means generating stylized images).<br />You can switch these two procedures by changing the flag `IS_TRAINING`.
- By default,<br />(1) The content images lie in the folder `"./images/content/"`<br />(2) The style images lie in the folder `"./images/style/"`<br />(3) The weights file of the pre-trained VGG-19 lies in the current working directory. (See `Prerequisites` below)<br />(4) The MS-COCO images dataset for training lies in the folder `"./MS_COCO/"` (See `Prerequisites` below)<br />(5) The checkpoint files of trained models lie in the folder `"./models/"` (You should create this folder manually before training)<br />(6) After inferring procedure, the stylized images will be generated and put in the folder `"./outputs/"`
- For training, you should make sure (2), (3), (4) and (5) are prepared correctly.
- For inferring, you should make sure (1) and (5) are prepared correctly.
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

