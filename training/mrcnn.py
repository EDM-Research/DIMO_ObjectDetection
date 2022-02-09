from mrcnn import config, utils, model as modellib
from training import augmentation
import os
import configparser

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
                epochs=100,
                layers='heads' if use_coco_weights else 'all',
                augmentation=augmenters,
                custom_callbacks=custom_callbacks)