from data.dimo_loader import DimoLoader
from pathlib import Path
from typing import List
import numpy as np
from bop_toolkit_lib import misc, visibility, inout, renderer
import os
import shutil

dimo_data = {
    'im_width': 2560,
    'im_height': 2048
}


def create_or_empty_folder(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def create_dimo_masks(path: str, subsets: List[str]) -> None:
    """
    Generates the visible mask for each object in each image.
    Masks are saved separately as a binary image for each object under {scene_id}/masks/{image_id}/{object_no}.png
    :param path: path to DIMO dataset
    :param subsets: subsets of dimo dataset to generate masks for (eg. sim_jaigo)
    """
    dimo_loader = DimoLoader()
    dimo_ds = dimo_loader.load(Path(path), cameras=subsets)

    for subset_name in subsets:
        subset = dimo_ds[subset_name]
        models = dimo_ds['models']

        for scene in subset:
            masks_path = os.path.join(scene['path'], 'masks/')
            print(f"Processing {scene['path']}")
            create_or_empty_folder(masks_path)
            ren = renderer.create_renderer(dimo_data['im_width'], dimo_data['im_height'], renderer_type='vispy', mode='depth')

            for model in models:
                ren.add_object(model['id'], model['cad'])
            for image in scene['images']:
                camera = image['camera']
                image_masks_path = os.path.join(masks_path, str(image['id']).zfill(6))
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
                    mask_visib = visibility.estimate_visib_mask_gt(dist_image, dist_map, 15, visib_mode='bop19') * 255
                    inout.save_im(object_mask_path, np.array(mask_visib, dtype=np.uint8))





