import json
import os
import numpy as np


class DimoLoader:

    def load(self, path, models_dir='models', cameras=['real_jaigo']):
        result = {}
        if models_dir:
            result['models'] = self.load_models(path / models_dir)
        for camera in cameras:
            result[camera] = self.load_scenes(path / camera)
        return result

    def load_models(self, path):
        with open(path / 'models_info.json') as f:
            models = json.load(f)
            result = []
            for model_id, model in models.items():
                model_id = int(model_id)  # convert id from str to int
                model['cad'] = path / f'obj_{int(model_id):06d}.ply'  # add cad path
                model['id'] = model_id
                result.append(model)
            return result

    def load_scenes(self, path):
        return [self.load_scene(path) for path in sorted(path.glob('[!.]*')) if os.path.isdir(path)]

    def load_scene(self, path):
        scene_id = int(path.name)
        result = {
            "id": scene_id,
            "path": path
        }
        images = []
        with open(path / 'scene_camera.json') as f_scene_camera, \
                open(path / 'scene_gt.json') as f_scene_gt:
            scene_camera = json.load(f_scene_camera)
            scene_gt = json.load(f_scene_gt)
            assert scene_camera.keys() == scene_gt.keys(), "labels are not consistent for all images"
            for image_id in scene_camera.keys():
                images.append(
                    self.load_image(path, int(image_id), scene_camera[image_id], scene_gt[image_id]))
        result['images'] = images
        return result

    def load_image(self, scene_path, image_id, camera, scene_gt):
        return {
            'id': image_id,
            'path': scene_path / 'rgb' / f'{int(image_id):06d}.png',
            'camera': self.load_camera(camera)
            'objects': self.load_objects(scene_gt)
        }

    def load_camera(self, camera):
        K = np.reshape(camera['cam_K'], (3, 3))
        T = self.load_pose(camera['cam_R_w2c'], camera['cam_t_w2c'])
        return {
            'K': K,
            'cam_2world': T
        }

    def load_objects(self, scene_gt):
        result = []
        for o in scene_gt:
            result.append({
                'id': int(o['obj_id']),
                'model_2cam': self.load_pose(o['cam_R_m2c'], o['cam_t_m2c']),
                'cam_R_m2c': o['cam_R_m2c'],
                'cam_t_m2c': o['cam_t_m2c']
            })
        return result

    def load_pose(self, R, t):
        T = np.eye(4, 4)
        T[:3, :3] = np.reshape(R, (3, 3))
        T[:3, 3] = t
        return T