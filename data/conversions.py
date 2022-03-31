import os, json
import cv2
from typing import List, Tuple
import numpy as np
from data.dimo_loader import DimoLoader
from pathlib import Path
from data.utils import create_or_empty_folder, get_file_count
import shutil


def get_bbox(img: np.array) -> Tuple[int, int, int, int]:
    if not np.any(img):
        return 0, 0, 0, 0

    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return x_min, x_max, y_min, y_max


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


def convert_to_new_mask_format(path: str, subsets: List[str]) -> None:
    dimo_loader = DimoLoader()
    dimo_ds = dimo_loader.load(Path(path), cameras=subsets)

    for subset_name in subsets:
        subset = dimo_ds[subset_name]
        for scene in subset:
            old_masks_path = os.path.join(scene['path'], 'masks/')
            new_masks_path = os.path.join(scene['path'], 'mask_visib/')

            if get_file_count(new_masks_path) == sum([len(image['objects']) for image in scene['images']]):
                continue

            print(f"Converting scene {scene['id']}")
            create_or_empty_folder(new_masks_path)

            for image in scene['images']:
                image_masks_path = os.path.join(old_masks_path, str(image['id']).zfill(6))
                for i in range(len(image['objects'])):
                    object_mask_path = os.path.join(image_masks_path, f"{i}.png")
                    shutil.move(object_mask_path, os.path.join(new_masks_path, f"{str(image['id']).zfill(6)}_{str(i).zfill(6)}.png"))

                os.rmdir(image_masks_path)
            os.rmdir(old_masks_path)


if __name__ == "__main__":
    convert_to_new_mask_format("F:/Data/dimo", ["real_jaigo_000-150",
                                                    "sim_jaigo_real_light_real_pose",
                                                    "sim_jaigo_rand_light_real_pose"]
                               )
