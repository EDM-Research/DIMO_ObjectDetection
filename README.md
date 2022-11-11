# DIMO Object Detection
Object detection for the DIMO dataset. Uses the Mask-RCNN model.

## Installation
First clone the repo using

    git clone https://github.com/EDM-Research/DIMO_ObjectDetection.git

 
 Install requirements using pip

     pip install -r requirements.txt

This repository contains modified versions:
- [BOP tookit](https://github.com/thodan/bop_toolkit)
- [DIMO Loader](https://github.com/pderoovere/dimo)
- [This modified version](https://github.com/akTwelve/Mask_RCNN) of [Matterport's MaskRCNN](https://github.com/matterport/Mask_RCNN).

## Dataset

Create a file named `config.ini` with the following contents:
```
[USER_SETTINGS]
dimo_path = **dimo path**
model_folder = **folder where trained models are saved**
images_per_gpu = 1 **increase this depending on GPU memory**
```
Download the [DIMO dataset](https://github.com/pderoovere/dimo) base and the desired subsets.
Set the `DIMO_PATH` in `config.ini` to the download location.

This repository also works with other datasets in BOP format.

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

The dimo path from the config file can be overwritten with the `--dimo_path` parameter. This is possible for all commands.
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
- `-t` The model will be initialized with weights pretrained on COCO. By default, only the heads will be trained, the ResNet layers are frozen.
- `--layers [layer_name]` Specify which layers of the model are trained. This can either be `all`, `heads`, `3+`, `4+` or `5+`, corresponding to the ResNet stages that will be trained. This defaults to `all`, unless the `-t` flag is supplied.
- `--save_all` Save the model after each epoch.
- `--model [model_id]` Specify the model id and training will resume from the last checkpoint of that model. The id is the name of the folder the checkpoints are saved in. Checkpoints are only available for models trained with `--save_all`.
- `--image_counts [count_1, count_2, ..]` Optionally specify the amount of images to use from each dataset, by default all images are used.
- `--ft_subsets [subset1, subset2, ..]` Optionally specify what subsets to finetune on. All layers are trained for 25 epochs at a lower LR during finetuning.
- `--ft_image_counts [count_1, count_2, ..]` Optionally specify how many images to use from the finetune datasets

Trained models will be placed in the `models_folder` from the config.

Training progress can be visualised with tensorboard:
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

## Citation
Use this bibtex to cite this repository:
```
@inproceedings{dimo_objectdetection_2022,
	title={Analysis of Training Object Detection Models with Synthetic Data},
	author={Vanherle Bram, Moonen Steven, Van Reeth Frank and Nick Michiels},
	year={2022},
	month={November},
	pages={},
	articleno={},
	numpages={},
	booktitle={Proceedings of the British Machine Vision Conference (BMVC)},
	publisher={BMVA Press},
	address = {},
	editor={},
}
```
