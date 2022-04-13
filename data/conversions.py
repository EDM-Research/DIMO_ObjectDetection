import os, json
import cv2
from typing import List, Tuple
import numpy as np
import skimage

from data.dimo_loader import DimoLoader
from pathlib import Path
from data.utils import create_or_empty_folder, get_file_count
import shutil
import mrcnn.utils


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


def shrink_dataset(path: str, target_path: str) -> None:
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        twin_file_path = os.path.join(target_path, file)
        if os.path.isdir(file_path):
            if file == "models":
                shutil.copytree(file_path, twin_file_path)
            else:
                create_or_empty_folder(twin_file_path)
                for subfile in os.listdir(file_path):
                    subfile_path = os.path.join(file_path, subfile)
                    twin_subfile_path = os.path.join(twin_file_path, subfile)
                    if os.path.isdir(subfile_path):
                        create_or_empty_folder(twin_subfile_path)

                        shutil.copyfile(os.path.join(subfile_path, "scene_camera.json"),
                                        os.path.join(twin_subfile_path, "scene_camera.json"))
                        shutil.copyfile(os.path.join(subfile_path, "scene_gt.json"),
                                        os.path.join(twin_subfile_path, "scene_gt.json"))
                        shutil.copyfile(os.path.join(subfile_path, "scene_gt_world.json"),
                                        os.path.join(twin_subfile_path, "scene_gt_world.json"))
                        shutil.copyfile(os.path.join(subfile_path, "scene_info.json"),
                                        os.path.join(twin_subfile_path, "scene_info.json"))

                        mask_path = os.path.join(subfile_path, "mask_visib")
                        rgb_path = os.path.join(subfile_path, "rgb")

                        twin_mask_path = os.path.join(twin_subfile_path, "mask_visib")
                        twin_rgb_path = os.path.join(twin_subfile_path, "rgb")

                        create_or_empty_folder(twin_rgb_path)
                        create_or_empty_folder(twin_mask_path)
                        for file in os.listdir(rgb_path):
                            rgb = skimage.io.imread(os.path.join(rgb_path, file))
                            small_rgb = mrcnn.utils.resize(rgb, (820, 1024), preserve_range=True).astype(np.uint8)
                            skimage.io.imsave(os.path.join(twin_rgb_path, file), small_rgb)

                        for file in os.listdir(mask_path):
                            mask = skimage.io.imread(os.path.join(mask_path, file))
                            small_mask = mrcnn.utils.resize(mask, (820, 1024), preserve_range=True).astype(np.uint8)
                            skimage.io.imsave(os.path.join(twin_mask_path, file), small_mask)

                    else:
                        shutil.copyfile(subfile_path, twin_subfile_path)


if __name__ == "__main__":
    shrink_dataset("C:/Users/bvanherle/Documents/Datasets/dimo", "C:/Users/bvanherle/Documents/Datasets/dimo_small")
