import configparser
import random

import cv2

from mrcnn import utils, visualize, config, model
from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import skimage


class DimoConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 4
    USE_MINI_MASK = True
    STEPS_PER_EPOCH = 1000
    TRAIN_ROIS_PER_IMAGE = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 8 + 1

    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        user_config = configparser.ConfigParser()
        user_config.read('config.ini')
        if 'images_per_gpu' in user_config['USER_SETTINGS'].keys():
            self.IMAGES_PER_GPU = int(user_config['USER_SETTINGS']['images_per_gpu'])
        super().__init__()


class DimoInferenceConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    USE_MINI_MASK = False

    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        super().__init__()


class DIMODataset(utils.Dataset):
    def load_dataset(self, path: str, subsets: List[str], split: str = "train", image_counts: list = None):
        assert split in ["train", "val", "test"]
        assert image_counts is None or len(image_counts) == len(subsets), "Image count needs to be specified for each subset"
        supported_extensions = ['png', 'jpg', 'jpeg']
        self.split = split
        self.subsets = subsets

        dimo_loader = DimoLoader()
        dimo_ds = dimo_loader.load(Path(path), cameras=subsets, models_dir=None)
        image_no = 0
        debug = os.environ.get("DEBUG_MODE", "0") == "1"

        model_to_id = {}

        for i, model in enumerate(dimo_ds['classes']):
            self.add_class("dimo", i + 1, str(model))
            model_to_id[str(model)] = i + 1

        for subset_id, subset_name in enumerate(subsets):
            subset = dimo_ds[subset_name]
            image_ids = self.get_image_ids(path, subset_name)
            if image_counts:
                random.seed(10)
                image_ids = random.sample(image_ids, min(image_counts[subset_id], len(image_ids)))
            for scene in subset:

                masks_path = os.path.join(scene['path'], 'mask_visib/')
                for image in scene['images']:
                    if f"{scene['id']}_{image['id']}" not in image_ids:
                        continue

                    instance_masks = []
                    instance_ids = []

                    one_indexed = True
                    for i, object in enumerate(image['objects']):
                        instance_ids.append(model_to_id[str(object['id'])])
                        image_path = os.path.join(masks_path, f"{str(image['id']).zfill(6)}_{str(i).zfill(6)}")
                        mask_path = self.find_file(image_path, supported_extensions)
                        if mask_path is not None:
                            instance_masks.append(mask_path)
                        elif i == 0:
                            one_indexed = True

                    if one_indexed:
                        image_path = os.path.join(masks_path, f"{str(image['id']).zfill(6)}_{str(len(image['objects'])).zfill(6)}")
                        mask_path = self.find_file(image_path, supported_extensions)
                        if mask_path is not None:
                            instance_masks.append(mask_path)

                    assert len(instance_masks) == len(instance_ids), "Number of masks does not match number of objects"

                    self.add_image(
                        "dimo",
                        image_id=f"{subset_name}_{scene['id']}_{image['id']}",
                        path=image['path'],
                        instance_masks=instance_masks,
                        instance_ids=instance_ids
                    )
                    image_no += 1

                    if debug and image_no > 20:
                        return

    def find_file(self, path: str, extensions: list):
        for extension in extensions:
            full_path = f"{path}.{extension}"
            if os.path.exists(full_path):
                return full_path
        return None

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        mask_0 = skimage.io.imread(image_info["instance_masks"][0])
        _, mask_0 = cv2.threshold(mask_0, 127,255, cv2.THRESH_BINARY)
        masks = np.zeros((mask_0.shape[0], mask_0.shape[1], len(image_info['instance_ids'])))
        masks[:, :, 0] = mask_0

        for object_no, mask_path in enumerate(image_info["instance_masks"][1:]):
            mask = skimage.io.imread(mask_path)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            masks[:, :, object_no+1] = mask
        return masks.astype(np.uint8), np.array(image_info["instance_ids"], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def get_image_ids(self, path: str, subset: str) -> List[str]:
        subset_path = os.path.join(path, subset)
        split_file_path = os.path.join(subset_path, f"{self.split}.txt")
        with open(split_file_path, 'r') as f:
            ids = [line.rstrip() for line in f]
        return ids


def get_dimo_datasets(path: str, subsets: List[str], train_image_counts: list = None) -> Tuple[DIMODataset, DIMODataset, DimoConfig]:
    dataset_train = DIMODataset()
    dataset_train.load_dataset(path, subsets, split="train", image_counts=train_image_counts)
    dataset_train.prepare()

    dataset_val = DIMODataset()
    dataset_val.load_dataset(path, subsets, split="val")
    dataset_val.prepare()

    config = DimoConfig(len(dataset_train.class_ids))

    return dataset_train, dataset_val, config


def get_test_dimo_dataset(path: str, subsets: List[str]) -> DIMODataset:
    dataset = DIMODataset()
    dataset.load_dataset(path, subsets, split="test")
    dataset.prepare()

    return dataset


def get_test_dimo_config(dataset: DIMODataset, model_id: str) -> DimoInferenceConfig:
    classes = len(dataset.class_ids)

    # very ugly hack, some models were trained with two extra classes
    if model_id in ["dimo20220212T1254"]:
        classes += 2

    config = DimoInferenceConfig(classes)

    return config


def get_dataset_images(dataset: DIMODataset, config: config.Config):
    return [model.load_image_gt(dataset, config, id)[0] for id in dataset.image_ids]