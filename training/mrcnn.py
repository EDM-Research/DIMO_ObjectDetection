from data.mrcnn_dimo import DIMODataset
from mrcnn import config, utils, model as modellib
from training import augmentation
import os
import configparser
from mrcnn.config import Config

COCO_WEIGHTS_PATH = 'weights/mask_rcnn_coco.h5'

import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def get_wandb_info():
    user_config = configparser.ConfigParser()
    user_config.read('config.ini')

    if 'wandb' not in user_config['USER_SETTINGS'].keys() or not bool(user_config['USER_SETTINGS']['wandb']):
        return None
    else:
        return user_config['USER_SETTINGS']['wandb_project'], user_config['USER_SETTINGS']['wandb_entity']


def get_model_folder():
    user_config = configparser.ConfigParser()
    user_config.read('config.ini')

    if 'model_folder' in user_config['USER_SETTINGS'].keys():
        return user_config['USER_SETTINGS']['model_folder']
    else:
        return 'models'


def train(train_set: DIMODataset, val_set: DIMODataset, config: config.Config, use_coco_weights: bool = True,
          augment: bool = True, checkpoint_model: modellib.MaskRCNN = None, ft_train_set: DIMODataset = None, layers: str = 'heads', save_all: bool = True):
    assert layers in ['3+', '4+', '5+', 'heads', 'all']

    augmenters = augmentation.augmenters if augment else None
    if checkpoint_model:
        model = checkpoint_model
    else:
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=get_model_folder())

    custom_callbacks = []

    wandb_info = get_wandb_info()
    if wandb_info:
        import wandb
        from wandb.keras import WandbCallback
        wandb_config = {
            'subsets': train_set.subsets,
            'model_id': model.log_dir.split("/")[-1],
            'learning_rate': config.LEARNING_RATE,
            'steps_per_epoch': config.STEPS_PER_EPOCH,
            'images_per_GPU': config.IMAGES_PER_GPU,
            'training_images': len(train_set.image_ids)
        }
        wandb.init(project=wandb_info[0], entity=wandb_info[1], config=wandb_config)
        custom_callbacks.append(WandbCallback())

    layers = layers if use_coco_weights else 'all'

    logging.info(f"Training model {model.log_dir}")
    logging.info(f"\tTraining on {'+'.join(train_set.subsets)}")
    logging.info(f"\tTraining images: {train_set.num_images}")
    logging.info(f"\tAugmentation: {augment})")
    logging.info(f"\tTransfer Learning: {use_coco_weights}\n")
    logging.info(f"\tTraining layers: {layers}\n")

    if use_coco_weights and checkpoint_model is None:
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    train_epochs = 75 if ft_train_set else 100
    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=train_epochs,
                layers=layers,
                augmentation=augmenters,
                custom_callbacks=custom_callbacks,
                save_all=save_all)

    if ft_train_set:
        logging.info(f"\tFinetuning on {'+'.join(ft_train_set.subsets)}")
        logging.info(f"\tFintuning on images: {ft_train_set.num_images}")
        model.train(ft_train_set, val_set,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='all',
                    augmentation=augmenters,
                    custom_callbacks=custom_callbacks,
                    save_all=save_all)

    logging.info("==========================================================================")


def get_epoch_no(file_name: str) -> int:
    return int(file_name.split('.')[0].split('_')[-1])


def get_available_epochs(model_dir: str) -> list:
    epochs = []

    for file in os.listdir(model_dir):
        if file.endswith('.h5'):
            epochs.append(get_epoch_no(file))

    return epochs


def get_file_for_epoch(model_dir: str, epoch: int = None) -> str:
    last_epoch = 0
    last_epoch_file = ""

    for file in os.listdir(model_dir):
        if file.endswith('.h5'):
            current_epoch = get_epoch_no(file)
            if current_epoch > last_epoch:
                last_epoch = current_epoch
                last_epoch_file = file
            if current_epoch == epoch:
                return file

    return last_epoch_file


def load_model(model_id: str, config: Config, epoch: int = None, mode: str = "inference") -> modellib.MaskRCNN:
    assert mode in ["training", "inference"], f"Mode can only be training or inference, not {mode}"
    model = modellib.MaskRCNN(mode=mode, config=config, model_dir=get_model_folder())
    model_file = get_file_for_epoch(f"{get_model_folder()}/{model_id}", epoch)
    model_path = f"{get_model_folder()}/{model_id}/{model_file}"
    print(f"Loading model from {model_path}")
    model.load_weights(model_path, by_name=True)
    return model