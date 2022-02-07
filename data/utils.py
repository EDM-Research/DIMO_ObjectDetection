import json
import random

from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List, Tuple
import numpy as np
from bop_toolkit_lib import misc, visibility, inout, renderer
import os
import shutil
import cv2

dimo_data = {
    'im_width': 2560,
    'im_height': 2048
}


def get_bbox(img: np.array) -> Tuple[int, int, int, int]:
    if not np.any(img):
        return 0, 0, 0, 0

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return x_min, x_max, y_min, y_max


def create_or_ignore_folder(path: str):
    if not os.path.exists(path):
        os.mkdir(path)


def create_or_empty_folder(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def get_file_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    else:
        return len(os.listdir(path))


def create_dimo_masks(path: str, subsets: List[str], override: bool = False) -> None:
    """
    Generates the visible mask for each object in each image.
    Masks are saved separately as a binary image for each object under {scene_id}/masks/{image_id}/{object_no}.png
    :param override: masks that are already generated are ignored if set to false, otherwise new masks are generated
    :param path: path to DIMO dataset
    :param subsets: subsets of dimo dataset to generate masks for (eg. sim_jaigo)
    """
    dimo_loader = DimoLoader()
    dimo_ds = dimo_loader.load(Path(path), cameras=subsets)

    for subset_name in subsets:
        subset = dimo_ds[subset_name]
        models = dimo_ds['models']

        ren = renderer.create_renderer(dimo_data['im_width'], dimo_data['im_height'], renderer_type='vispy', mode='depth')
        for model in models:
            ren.add_object(model['id'], model['cad'])

        for scene in subset:
            masks_path = os.path.join(scene['path'], 'masks/')
            print(f"Processing {scene['path']}")

            if override:
                create_or_empty_folder(masks_path)
            else:
                create_or_ignore_folder(masks_path)

            for image in scene['images']:
                camera = image['camera']
                image_masks_path = os.path.join(masks_path, str(image['id']).zfill(6))

                if get_file_count(image_masks_path) == len(image['objects']):
                    continue

                create_or_empty_folder(image_masks_path)
                K = camera['K']
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

                dist_image = np.matrix(np.ones((dimo_data['im_height'], dimo_data['im_width'])) * np.inf)
                distance_maps = []
                for object in image['objects']:
                    depth_gt = ren.render_object(object['id'], np.array(object['cam_R_m2c']).reshape(3,3), np.array(object['cam_t_m2c']), fx, fy, cx, cy)['depth']
                    dist_gt = misc.depth_im_to_dist_im_fast(depth_gt, K)

                    # set zero depths to infinity to compute closest object for total depth map
                    dist_gt[dist_gt == 0] = np.inf
                    dist_image = np.minimum(dist_image, dist_gt)

                    dist_gt[dist_gt == np.inf] = 0
                    distance_maps.append(dist_gt)

                dist_image[dist_image == np.inf] = 0

                for object_no, dist_map in enumerate(distance_maps):
                    object_mask_path = os.path.join(image_masks_path, f"{object_no}.png")
                    mask_visib = visibility.estimate_visib_mask_gt(dist_image, dist_map, 1, visib_mode='bop19') * 255
                    inout.save_im(object_mask_path, np.array(mask_visib, dtype=np.uint8))


def create_dimo_train_split(path: str, subsets: List[str], train: float = 0.9, val: float = 0.05, test: float = 0.05, seed: int = None) -> None:
    """
    Given the path to a dimo dataset, this function will split the scenes of the given subsets in training, validation
    and testing dataset. The function creates files of name {split}.txt
    :param path:    path to the dimo dataset
    :param subsets: subsets to generate split for
    :param train:   portion of scenes to be training dataset
    :param val:     portion of scenes to be validation dataset
    :param test:    portion of scenes to be test dataset
    :param seed:    optionally set random seed
    :return:
    """
    def write_to_file(file: str, data: list):
        with open(file, 'w') as f:
            for element in data:
                f.write(f"{element}\n")

    if seed:
        random.seed(seed)

    train /= sum([train, val, test])
    test /= sum([train, val, test])
    val /= sum([train, val, test])

    dimo_loader = DimoLoader()
    dimo_ds = dimo_loader.load(Path(path), cameras=subsets)

    for subset_name in subsets:
        subset = dimo_ds[subset_name]
        scene_ids = [scene['id'] for scene in subset]
        random.shuffle(scene_ids)

        train_ids = scene_ids[:int(train * len(scene_ids))]
        val_ids = scene_ids[int(train * len(scene_ids)):int((val + train) * len(scene_ids))]
        test_ids = scene_ids[int((val + train) * len(scene_ids)):]

        subset_path = os.path.join(path, f"{subset_name}/")
        write_to_file(os.path.join(subset_path, "train.txt"), train_ids)
        write_to_file(os.path.join(subset_path, "val.txt"), val_ids)
        write_to_file(os.path.join(subset_path, "test.txt"), test_ids)


def dimo_to_createml(path: str):
    annotations = []

    scenes = [scene for scene in os.listdir(path) if os.path.isdir(os.path.join(path, scene))]

    for scene in scenes:
        scene_path = os.path.join(path, scene)
        masks_path = os.path.join(scene_path, 'mask_visib/')

        with open(os.path.join(scene_path, "scene_gt.json"), 'r') as f:
            scene_dict = json.load(f)
            for image in scene_dict.keys():
                annotation_dict = {
                    "image": f"{scene.zfill(6)}/rgb/{image.zfill(6)}.jpg",
                    "annotations": []
                }
                for object in scene_dict[image]:
                    mask_file = f"{image.zfill(6)}_{str(object['obj_id']).zfill(6)}.jpg"
                    mask_path = os.path.join(masks_path, mask_file)
                    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    x_min, x_max, y_min, y_max = get_bbox(mask_image)
                    annotation_dict["annotations"].append({
                        "label": str(object['obj_id']),
                        "coordinates": {
                            "x": int(x_min),
                            "y": int(y_min),
                            "width": int(x_max - x_min),
                            "height": int(y_max - y_min)
                        }
                    })
                annotations.append(annotation_dict)

    with open(os.path.join(path, "createml.json"), 'w') as f:
        json.dump(annotations, f)


if __name__ == "__main__":
    dimo_to_createml("E:/projects/renderings/deo/bop/train_PBR/")