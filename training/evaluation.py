from mrcnn.utils import Dataset, compute_ap
from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np


class DimoInferenceConfig(Config):
    NAME = "dimo"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 8 + 1     # 8 models + background
    DETECTION_MIN_CONFIDENCE = 0.5


def load_model(model_dir: str) -> modellib.MaskRCNN:
    model = modellib.MaskRCNN(mode="inference", config=DimoInferenceConfig(), model_dir=f"models/{model_dir}")
    model.load_weights(model.find_last())
    return model


def get_detections(dataset: Dataset, model: modellib.MaskRCNN, config: Config) -> list:
    results = []

    for image_id in dataset.image_ids:
        image, *_ = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        result = model.detect([image], verbose=0)[0]
        result['image_id'] = image_id
        results.append(result)

    return results


def compute_map(results: list, dataset: Dataset, config: Config, iou_threshold: float = 0.5) -> float:
    aps = []

    for result in results:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, result['image_id'],
                                                                                  use_mini_mask=False)
        ap, precisions, recalls, overlaps = compute_ap(
            gt_boxes=gt_bbox,
            gt_class_ids=gt_class_id,
            gt_masks=gt_mask,
            pred_boxes=result['rois'],
            pred_class_ids=result['class_ids'],
            pred_scores=result['scores'],
            pred_masks=result['masks'],
            iou_threshold=iou_threshold
        )

        aps.append(ap)

    return np.mean(aps)
