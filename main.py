import skimage
import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import utils as mrcnn_utils
from mrcnn import visualize as mrcnn_visualise
from mrcnn import model as modellib
from training import evaluation
from utils import visualize, interactions
import configparser
import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

config = configparser.ConfigParser()
config.read('config.ini')

DIMO_PATH = config['USER_SETTINGS']['dimo_path']


def train_subsets(subsets: list, model_id: str=None, augment: bool = False, transfer_learning: bool = False):
    from training import mrcnn

    train, val, config = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)

    model = evaluation.load_model(model_id, config, mode="training") if model_id else None

    print(f"training images: {len(train.image_ids)}")
    print(f"validation images: {len(val.image_ids)}")

    mrcnn.train(train, val, config, augment=augment, use_coco_weights=transfer_learning, checkpoint_model=model)


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


def test_subsets(subsets: list, model_id: str):
    iou = 0.5
    dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)

    model = evaluation.load_model(model_id, config)
    results = evaluation.get_detections_dataset(dataset, model, config)
    map = evaluation.compute_map(results, dataset, config, iou)
    precision, recall = evaluation.compute_mean_pand(results, dataset, config, iou)

    print(f"maP @ iou = {iou} = {map}")
    print(f"precision @ iou = {iou} = {precision}")
    print(f"recall @ iou = {iou} = {recall}")

    evaluation.show_results(results, dataset, config)


def test_folder(folder: str,  model_id: str, num_classes: int, select_roi=False, save_folder=None):
    config = data.mrcnn_dimo.DimoInferenceConfig(num_classes=num_classes)
    model = evaluation.load_model(model_id, config)

    images = [cv2.imread(os.path.join(folder, file)) for file in os.listdir(folder) if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")]

    rois = interactions.select_rois(images) if select_roi else None

    for i, image in enumerate(images):
        input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if rois:
            r = rois[i]
            input_image = input_image[r[0]:r[1], r[2]: r[3]]
        result = model.detect([input_image])[0]

        plot = visualize.render_instances(
            image=input_image,
            boxes=result['rois'],
            masks=result['masks'],
            class_ids=result['class_ids'],
            class_names=[str(i) for i in range(num_classes)],
            scores=result['scores']
        )

        plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)

        if save_folder:
            cv2.imwrite(os.path.join(save_folder, f"{i}.png"), plot)
        else:
            cv2.imshow("Result", plot)
            cv2.waitKey(0)


if __name__ == "__main__":
    test_subsets(["debug"], "deo_002")
