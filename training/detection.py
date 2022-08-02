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


def get_umap(dataset: Dataset, model: modellib.MaskRCNN, config: Config, level: int = 0) -> np.array:
    assert 0 <= level <= 3
    umap_train_sample = 5
    total_samples = 10

    feature_detector = model.get_feature_detector()
    features = []

    ids = dataset.image_ids
    np.random.shuffle(ids)

    for i, image_id in enumerate(ids[:umap_train_sample]):
        print(f"\rTraining on image {i}/{umap_train_sample}", end='')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        molded_images, image_metas, windows = model.mold_inputs([image])
        image_shape = molded_images[0].shape
        anchors = model.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        output = feature_detector([molded_images, image_metas, anchors])[level]

        features.append(output.flatten())

    features = np.array(features)
    reducer = umap.UMAP()
    reducer.fit(features)

    features = []
    for i, image_id in enumerate(ids[umap_train_sample:total_samples]):
        print(f"Computing for image {i}/{total_samples - umap_train_sample}", end='\r')

        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        molded_images, image_metas, windows = model.mold_inputs([image])
        image_shape = molded_images[0].shape
        anchors = model.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        output = feature_detector([molded_images, image_metas, anchors])[level].flatten()
        features.append(output.flatten())

    embedding = reducer.transform(np.array(features))

    embeddings = np.concatenate((reducer.embedding_, embedding))

    return embeddings