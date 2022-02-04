from mrcnn import config, utils, model as modellib
from training import augmentation
import os

COCO_WEIGHTS_PATH = 'weights/mask_rcnn_coco.h5'


def train(train_set: utils.Dataset, val_set: utils.Dataset, config: config.Config, use_coco_weights: bool = True, augment: bool = True):
    augmenters = augmentation.augmenters if augment else None
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir='models')

    print(f"Saving model to {model.log_dir}\n")

    if use_coco_weights:
        weights_path = COCO_WEIGHTS_PATH
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
        model.load_weights(weights_path, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

    model.train(train_set, val_set,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers='heads' if use_coco_weights else 'all',
                augmentation=augmenters)