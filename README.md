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

Create a file named `config.ini` with the following contents:
```
[USER_SETTINGS]
dimo_path = **dimo path**
```
Download the [DIMO dataset](https://github.com/pderoovere/dimo) base and the desired subsets.
Set the `DIMO_PATH` in `config.ini` to the download location.

## Use

### Preparing data
Before training a model the segmentation masks need to be generated and a train-val-test split needs to be made. This can be done using the following command for an arbitrary number of subsets:
```
python dimo_detection.py prepare --subsets subset1 subsets2
```
`subset` should be the name of a subfolder in your `dimo`folder
The following flags can be used:
- `-o` override existing masks if there are found
- `-s` If this flag is set, all images of the same scene will be placed in the same set.

### Inspecting data
To inspect one or more datasets you can use the following command:
```
python dimo_detection.py show --subsets subset1 subsets2
```

### Training
To train a model for one or more subsets you can use the following command:
```
python dimo_detection.py train --subsets subset1 subsets2
```
The following flags can be used:
- `-a` If this flag is set data augmentations will be applied during training. These are found in `training/augmentation.py`
- `-t` The model will be initialized with weights pretrained on COCO. Only the heads will be trained, the ResNet layers are frozen.
- `--model` Specify the model id and training will resume from the last checkpoint of that model. The id is the name of the folder the checkpoints are saved in.
Trained models will be placed in the `models` folder. Training progress can be visualised with tensorboard:
```
tensorboard --logdir models/
```

#### Weights and Biases
Optionally the WandB platform can be used for logging.
To use this add the following information to the `config.ini` file under `[USER_SETTINGS]`.
```
wandb = True
wandb_project = **project_name**
wandb_entity = **user_name**
```

### Evaluation
A model can be evaluated on a test set with the following command. The mAP value is computed and images and results from the test set are shown.
```
python dimo_detection.py test --subsets subset1 subsets2 --model model
```