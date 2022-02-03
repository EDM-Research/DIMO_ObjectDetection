# DIMO Object Detection
Object detection for the DIMO dataset. Uses the Mask-RCNN model.

## Installation
First clone the repo using

    git clone https://github.com/EDM-Research/DIMO_ObjectDetection.git

 
 Install requirements using pip

     pip install -r requirements.txt

This repository contains modified versions of [BOP tookit](https://github.com/thodan/bop_toolkit) and [DIMO Loader](https://github.com/pderoovere/dimo), these should not be downloaded separately.

To train and run object detection [Matterport's MaskRCNN](https://github.com/matterport/Mask_RCNN) is used. The original version only supports Tensorflow 1.x.
If you have a more recent GPU with the Ampere architecture (GTX 3090) you can not use TF 1.x as does not support cuda 11.
If you are running tf 2.x you can use [this modified version](https://github.com/akTwelve/Mask_RCNN) of Matterport MaskRCNN.

To install either of those versions:
- Download the MaskRCNN repository
- Place the `maskrcnn` folder in the root of  this repository

## Dataset

Download the [DIMO dataset](https://github.com/pderoovere/dimo) base and the desired subsets. Set the `DIMO_PATH` in `main.py` to the download location.

## Use

Before training a model the segmentation masks need to be generated. This can be done using the following command for an arbitrary number of subsets:
```
python dimo_detection.py prepare --subsets subset1 subsets2
```
`subset` should be the name of a subfolder in your `dimo`folder

To inspect one or more datasets you can use the following command:
```
python dimo_detection.py show_subsets --subsets subset1 subsets2
```

To train a model for one or more subsets you can use the following command:
```
python dimo_detection.py train_subsets --subsets subset1 subsets2
```
Trained models will be placed in the `models` folder. Training progress can be visualised with tensorboard:
```
tensorboard --logdir models/
```
