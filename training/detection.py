import numpy as np

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


def get_feature_maps(dataset: Dataset, model: modellib.MaskRCNN, config: Config, level: int = 0) -> list:
    assert 0 <= level <= 3
    feature_detector = model.get_feature_detector()
    features = []

    for i, image_id in enumerate(dataset.image_ids):
        print(f"Testing image {i}/{len(dataset.image_ids)}", end='\r')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        molded_images, image_metas, windows = model.mold_inputs([image])
        image_shape = molded_images[0].shape
        anchors = model.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        output = feature_detector([molded_images, image_metas, anchors])[level]

        features.append(output.flatten())

    return features