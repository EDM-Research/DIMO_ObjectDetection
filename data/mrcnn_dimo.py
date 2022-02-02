import random

from mrcnn import utils, visualize, config
from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
import skimage


class DimoConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 8 + 1     # 8 models + background
    STEPS_PER_EPOCH = 1000
    DETECTION_MIN_CONFIDENCE = 0.9


class DIMODataset(utils.Dataset):
    def load_dataset(self, path: str, subsets: List[str], split: str = "train"):
        assert split in ["train", "val", "test"]
        self.split = split

        dimo_loader = DimoLoader()
        dimo_ds = dimo_loader.load(Path(path), cameras=subsets)
        image_no = 0
        debug = os.environ.get("DEBUG_MODE", "0") == "1"

        for model in dimo_ds['models']:
            self.add_class("dimo", int(model['id']), str(model['id']))

        for subset_name in subsets:
            subset = dimo_ds[subset_name]
            scene_ids = self.get_scene_ids(path, subset_name)
            for scene in subset:
                if scene['id'] not in scene_ids:
                    continue
                masks_path = os.path.join(scene['path'], 'masks/')
                for image in scene['images']:
                    image_masks_path = os.path.join(masks_path, str(image['id']).zfill(6))
                    instance_ids = [obj['id'] for obj in image['objects']]
                    image_data = skimage.io.imread(image['path'])
                    height, width = image_data.shape[:2]

                    self.add_image(
                        "dimo",
                        image_id=f"{subset_name}_{scene['id']}_{image['id']}",
                        path=image['path'],
                        masks_path=image_masks_path,
                        instance_ids=instance_ids,
                        width=width,
                        height=height
                    )
                    image_no += 1

                    if debug and image_no > 20:
                        return

    def load_mask(self, image_id):
        image_info = self.image_info[image_id]
        masks = np.zeros((image_info['height'], image_info['width'], len(image_info['instance_ids'])))
        object_count = 0
        for file in os.listdir(image_info["masks_path"]):
            if file.endswith(".png"):
                masks[:, :, object_count] = skimage.io.imread(os.path.join(image_info["masks_path"], file))
                object_count += 1
        return masks.astype(np.uint8), np.array(image_info["instance_ids"], dtype=np.int32)

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']

    def get_scene_ids(self, path: str, subset: str) -> List[int]:
        subset_path = os.path.join(path, subset)
        split_file_path = os.path.join(subset_path, f"{self.split}.txt")
        with open(split_file_path, 'r') as f:
            ids = [int(line.rstrip()) for line in f]
        return ids


def get_dimo_datasets(path: str, subsets: List[str]) -> Tuple[DIMODataset, DIMODataset, DimoConfig]:
    dataset_train = DIMODataset()
    dataset_train.load_dataset(path, subsets, split="train")
    dataset_train.prepare()

    dataset_val = DIMODataset()
    dataset_val.load_dataset(path, subsets, split="val")
    dataset_val.prepare()
    return dataset_train, dataset_val, DimoConfig()
