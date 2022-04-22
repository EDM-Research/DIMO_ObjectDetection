from mrcnn import config, utils, model as modellib
from training import augmentation
import os
import configparser
from mrcnn.config import Config

COCO_WEIGHTS_PATH = 'weights/mask_rcnn_coco.h5'


def get_wandb_info():
    user_config = configparser.ConfigParser()
    user_config.read('config.ini')

    if 'wandb' not in user_config['USER_SETTINGS'].keys() or not bool(user_config['USER_SETTINGS']['wandb']):
        return None
    else:
        return user_config['USER_SETTINGS']['wandb_project'], user_config['USER_SETTINGS']['wandb_entity']


def train(train_set: utils.Dataset, val_set: utils.Dataset, config: config.Config, use_coco_weights: bool = True, augment: bool = True, checkpoint_model: modellib.MaskRCNN = None):
    augmenters = augmentation.augmenters if augment else None
    if checkpoint_model:
        model = checkpoint_model
    else:
        model = modellib.MaskRCNN(mode="training", config=config, model_dir='models')

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

    print(f"Saving model to {model.log_dir}\n")
    print(f"\nAugmentation: {augment}\t Transfer Learning: {use_coco_weights}\n")

    if use_coco_weights:
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=400,
                layers='heads' if use_coco_weights else 'all',
                augmentation=augmenters,
                custom_callbacks=custom_callbacks)


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
    model = modellib.MaskRCNN(mode=mode, config=config, model_dir=f"models")
    model_file = get_file_for_epoch(f"models/{model_id}", epoch)
    model_path = f"models/{model_id}/{model_file}"
    print(f"Loading model from {model_path}")
    model.load_weights(model_path, by_name=True)
    return model