import data.mrcnn_dimo
from data import utils as data_utils
from data import mrcnn_dimo
import os, random
from mrcnn import utils, visualize
from training import evaluation

DIMO_PATH = "D:/Datasets/DIMO/dimo"


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
    dataset_train, dataset_val, _ = mrcnn_dimo.get_dimo_datasets(DIMO_PATH, subsets)
    print(f"training images: {len(dataset_train.image_ids)}")
    print(f"validation images: {len(dataset_val.image_ids)}")

    while True:
        image_id = random.choice(dataset_train.image_ids)
        image = dataset_train.load_image(image_id)
        mask, class_ids = dataset_train.load_mask(image_id)
        # Compute Bounding box
        bbox = utils.extract_bboxes(mask)
        visualize.display_instances(image, bbox, mask, class_ids, dataset_train.class_names)


def test_subsets(subsets, model_dir):
    iou = 0.5
    dataset, config = data.mrcnn_dimo.get_test_dimo_dataset(DIMO_PATH, subsets)
    model = evaluation.load_model(model_dir, config)
    results = evaluation.get_detections(dataset, model)
    map = evaluation.compute_map(results, dataset, config, iou)

    print(f"maP @ iou = {iou} = {map}")

    evaluation.show_results(results, dataset)


if __name__ == "__main__":
    os.environ["DEBUG_MODE"] = "1"
    #prepare_subsets(["sim_jaigo_real_light_rand_pose"])
    train_subsets(["sim_jaigo_real_light_real_pose"])
