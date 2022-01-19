import random

from mrcnn import utils, visualize, config
from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List
import os
import numpy as np
import skimage


class DimoConfig(config.Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 8 + 1 # 8 models + background
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9


class DIMODataset(utils.Dataset):
    def load_dataset(self, path: str, subsets: List[str]):
        # TODO: Implement test/train/val set
        dimo_loader = DimoLoader()
        dimo_ds = dimo_loader.load(Path(path), cameras=subsets)
        image_no = 0
        debug = os.environ.get("DEBUG_MODE", "0") == "1"

        for model in dimo_ds['models']:
            self.add_class("dimo", int(model['id']), str(model['id']))

        for subset_name in subsets:
            subset = dimo_ds[subset_name]
            for scene in subset:
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
        return masks.astype(bool), np.array(image_info["instance_ids"])

    def image_reference(self, image_id):
        return self.image_info[image_id]['path']


if __name__ == "__main__":

    dataset = DIMODataset()
    dataset.load_dataset("F:/Data/dimo", ["sim_jaigo"])
    dataset.prepare()

    while True:
        image_id = random.choice(dataset.image_ids)
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)
        visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)
