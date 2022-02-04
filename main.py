import cv2.cv2

import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import utils, visualize
from mrcnn import model as modellib
from training import evaluation
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read('config.ini')

DIMO_PATH = config['USER_SETTINGS']['dimo_path']


def train_subsets(subsets):
    from training import mrcnn

    train, val, _ = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)

    print(f"training images: {len(train.image_ids)}")
    print(f"validation images: {len(val.image_ids)}")

    mrcnn.train(train, val, mrcnn_dimo.DimoConfig(), False)


def prepare_subsets(subsets):
    data_utils.create_dimo_masks(DIMO_PATH, subsets)
    data_utils.create_dimo_train_split(DIMO_PATH, subsets, seed=10)


def show_subsets(subsets):
    dataset_train, dataset_val, config = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)
    print(f"training images: {len(dataset_train.image_ids)}")
    print(f"validation images: {len(dataset_val.image_ids)}")

    while True:
        image_id = random.choice(dataset_train.image_ids)
        image_info = dataset_train.image_info[image_id]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train, config, image_id)
        # Compute Bounding box
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, title=image_info['id'])


def test_subsets(subsets, model_id):
    iou = 0.5
    dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)
    model = evaluation.load_model(model_id, config)
    results = evaluation.get_detections(dataset, model, config)
    map = evaluation.compute_map(results, dataset, config, iou)

    print(f"maP @ iou = {iou} = {map}")

    evaluation.show_results(results, dataset, config)


if __name__ == "__main__":
    os.environ["DEBUG_MODE"] = "0"
    show_subsets(["real_jaigo_000-150"])
