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


def get_umap(dataset: Dataset, model: modellib.MaskRCNN, config: Config, level: int = 0) -> np.array:
    assert 0 <= level <= 3
    umap_train_sample = 200
    total_samples = 1755

    feature_detector = model.get_feature_detector()
    features = []

    ids = dataset.image_ids
    np.random.shuffle(ids)

    print("Computing features for training")
    for i, image_id in enumerate(ids[:umap_train_sample]):
        print(f"\rComputing image {i}/{len(ids[:umap_train_sample])}", end='')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        output = run_feature_detector(feature_detector, model, image)[level]

        features.append(output.flatten())

    print("\nTraining model")
    features = np.array(features)
    reducer = umap.UMAP()
    reducer.fit(features)

    embeddings = reducer.embedding_

    print("Computing embeddings for other images")
    batch_count = math.ceil(len(ids[umap_train_sample:total_samples]) / umap_train_sample)
    for batch_no in range(batch_count):
        print(f"Computing features for batch {batch_no}")
        features = []

        for i, image_id in enumerate(ids[umap_train_sample + batch_no * umap_train_sample:min(umap_train_sample + (batch_no + 1) * umap_train_sample, len(ids))]):
            image, *_ = modellib.load_image_gt(dataset, config, image_id)
            print(f"\rComputing image {i}/{len(ids[umap_train_sample + batch_no * umap_train_sample:min(umap_train_sample + (batch_no + 1) * umap_train_sample, len(ids))])}", end='')
            output = run_feature_detector(feature_detector, model, image)[level]
            features.append(output.flatten())

        print(f"Computing embeddings for batch {batch_no}")
        embedding = reducer.transform(np.array(features))

        embeddings = np.concatenate((embedding, embeddings))

    return embeddings