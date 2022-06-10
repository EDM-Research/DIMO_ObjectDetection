import skimage
from matplotlib import pyplot as plt

import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import utils as mrcnn_utils
from mrcnn import visualize as mrcnn_visualise
from mrcnn import model as modellib
from training import evaluation, detection
from utils import visualize, interactions, file_io
import configparser
import cv2
import numpy as np
from training import mrcnn as mrcnn_training


config = configparser.ConfigParser()
config.read('config.ini')

DIMO_PATH = config['USER_SETTINGS']['dimo_path']


def test_batch(batch_file: str):
    model_tests = file_io.read_test_batch(batch_file)

    for test in model_tests:
        iou = 0.5
        dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, [test.test_subset])

        model = mrcnn_training.load_model(test.model_id, config)
        results = detection.get_detections_dataset(dataset, model, config)
        map = evaluation.compute_map(results, dataset, config, iou)
        precision, recall = evaluation.compute_mean_pand(results, dataset, config, iou)

        test.metrics = {
            "map": map,
            "precision": precision,
            "recall": recall
        }

    filename = f"{batch_file.split('.')[0]}_results.csv"
    file_io.write_test_metrics(model_tests, filename)


def train_subsets(subsets: list, model_id: str = None, augment: bool = False, transfer_learning: bool = False,
                  train_image_counts: list = None, ft_subsets: list = None, ft_image_count: int = None):
    # load training set
    train, val, config = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets, train_image_counts=train_image_counts)

    print(f"training images: {len(train.image_ids)}")
    print(f"validation images: {len(val.image_ids)}")

    # if specified, load fintuning dataset
    ft_train = None
    if ft_subsets:
        ft_train, _, _ = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, ft_subsets, train_image_counts=ft_image_count)

        print(f"finetuning images: {len(ft_train.image_ids)}")

    # load model to continue training, if specified
    model = mrcnn_training.load_model(model_id, config, mode="training") if model_id else None
    # train model
    mrcnn_training.train(train, val, config, augment=augment, use_coco_weights=transfer_learning, checkpoint_model=model, ft_train_set=ft_train)


def prepare_subsets(subsets: list, override: bool = False, split_scenes: bool = False):
    #data_utils.create_dimo_masks(DIMO_PATH, subsets, override=override)
    data_utils.create_dimo_train_split(DIMO_PATH, subsets, seed=10, split_scenes=split_scenes)


def show_subsets(subsets: list):
    dataset_train, dataset_val, config = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)
    config.USE_MINI_MASK = False

    print(f"training images: {len(dataset_train.image_ids)}")
    print(f"validation images: {len(dataset_val.image_ids)}")

    while True:
        image_id = random.choice(dataset_train.image_ids)
        image_info = dataset_train.image_info[image_id]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_train, config, image_id)
        # Compute Bounding box
        mrcnn_visualise.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, title=image_info['id'])


def test_subsets(subsets: list, model_id: str, save_results: bool = False):
    iou = 0.5
    dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)

    model = mrcnn_training.load_model(model_id, config)
    results = detection.get_detections_dataset(dataset, model, config)
    map = evaluation.compute_map(results, dataset, config, iou)
    precision, recall = evaluation.compute_mean_pand(results, dataset, config, iou)

    print(f"maP @ iou = {iou} = {map}")
    print(f"precision @ iou = {iou} = {precision}")
    print(f"recall @ iou = {iou} = {recall}")

    if save_results:
        visualize.save_results(results, mrcnn_dimo.get_dataset_images(dataset, config), "results/", dataset.class_names)
    else:
        visualize.show_results(results, mrcnn_dimo.get_dataset_images(dataset, config), dataset.class_names)


def test_folder(folder: str,  model_id: str, num_classes: int, select_roi=False, save_folder=None):
    config = data.mrcnn_dimo.DimoInferenceConfig(num_classes=num_classes)
    model = mrcnn_training.load_model(model_id, config)

    images = [cv2.cvtColor(cv2.imread(os.path.join(folder, file)),cv2.COLOR_BGR2RGB) for file in os.listdir(folder) if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]
    class_names = [str(i) for i in range(num_classes)]

    if select_roi:
        rois = interactions.select_rois(images)
        images = [image[r[0]:r[1], r[2]: r[3]] for r, image in zip(rois, images)]

    results = detection.get_detections_images(images, model)

    if save_folder:
        visualize.save_results(results, images, save_folder, class_names)
    else:
        visualize.show_results(results, images, class_names)


def test_epochs(subsets: list, models: list):
    test_frequency = 50
    iou = 0.5
    dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)

    results_dict = {}

    for model_id in models:
        available_epochs = np.array(mrcnn_training.get_available_epochs(f"{mrcnn_training.get_model_folder()}/{model_id}"))
        test_epochs = np.arange(0, np.max(available_epochs) + 1, test_frequency).astype(np.int)
        test_epochs[0] += 1
        tested_epochs = []
        maps = []
        for epoch in test_epochs:
            if epoch in available_epochs:
                model = mrcnn_training.load_model(model_id, config, epoch)
                results = detection.get_detections_dataset(dataset, model, config)
                map = evaluation.compute_map(results, dataset, config, iou)

                tested_epochs.append(epoch)
                maps.append(map)

        results_dict[model_id] = {
            'tested_epochs': tested_epochs,
            'maps': maps
        }

    with plt.style.context('Solarize_Light2'):
        for model_id in models:
            plt.plot(results_dict[model_id]['tested_epochs'], results_dict[model_id]['maps'], label = model_id)

        plt.legend()
        plt.show()
        plt.xlabel("Epoch")
        plt.ylabel("mAP")


if __name__ == "__main__":
    test_folder("C:/Users/bvanherle/Documents/Datasets/deo/1-4DPS instruments/", "deo_005", 5, save_folder="C:/Users/bvanherle/Documents/Datasets/deo/results/005")