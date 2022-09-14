import math

import numpy as np
import umap

from mrcnn.utils import Dataset
from mrcnn.config import Config
import mrcnn.model as modellib


def get_detections_dataset(dataset: Dataset, model: modellib.MaskRCNN, config: Config) -> list:
    results = []

    for i, image_id in enumerate(dataset.image_ids):
        print(f"Testing image {i}/{len(dataset.image_ids)}", end='\r')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        result = model.detect([image], verbose=0)[0]
        result['image_id'] = image_id
        results.append(result)

    return results


def get_detections_images(images: list, model: modellib.MaskRCNN) -> list:
    results = []

    for i, image in enumerate(images):
        print(f"Testing image {i}/{len(images)}", end='\r')
        result = model.detect([image], verbose=0)[0]
        result['image_id'] = i
        results.append(result)

    return results


def run_feature_detector(feature_detector, model, image):
    molded_images, image_metas, windows = model.mold_inputs([image])
    image_shape = molded_images[0].shape
    anchors = model.get_anchors(image_shape)
    anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

    output = feature_detector([molded_images, image_metas, anchors])

    return output
