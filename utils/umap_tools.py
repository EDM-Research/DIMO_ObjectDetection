import numpy as np
import umap
import math

from mrcnn.config import Config
from mrcnn.utils import Dataset
import mrcnn.model as modellib
from training.detection import run_feature_detector


def get_reducer(dataset: Dataset, model: modellib.MaskRCNN, config: Config, level: int = 0) -> umap.UMAP:
    assert 0 <= level <= 3
    samples = 5

    feature_detector = model.get_feature_detector()
    features = []

    ids = dataset.image_ids
    np.random.shuffle(ids)

    print("Computing features for training")
    for i, image_id in enumerate(ids[:samples]):
        print(f"\rComputing image {i}/{len(ids[:samples])}", end='')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        output = run_feature_detector(feature_detector, model, image)[level]

        features.append(output.flatten())

    print("\nTraining model")
    features = np.array(features)
    reducer = umap.UMAP(random_state=10)
    reducer.fit(features)

    return reducer


def reduce_dimension(dataset: Dataset, reducer: umap.UMAP, model: modellib.MaskRCNN, config: Config, level: int = 0) -> np.array:
    assert 0 <= level <= 3

    feature_detector = model.get_feature_detector()
    batch_size = 100

    ids = dataset.image_ids
    np.random.shuffle(ids)

    print("Computing embeddings for other images")
    batch_count = math.ceil(len(ids) / batch_size)

    embeddings = None
    for batch_no in range(batch_count):
        print(f"Computing features for batch {batch_no}/{batch_count}")
        features = []

        for i, image_id in enumerate(ids[batch_no * batch_size:min(batch_size * (batch_no + 1), len(ids))]):
            image, *_ = modellib.load_image_gt(dataset, config, image_id)
            output = run_feature_detector(feature_detector, model, image)[level]
            features.append(output.flatten())

        print(f"Computing embeddings for batch {batch_no}")
        embedding = reducer.transform(np.array(features))

        embeddings = embedding if embeddings is None else np.concatenate((embedding, embeddings))

    return embeddings
