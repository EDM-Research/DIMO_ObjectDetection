from mrcnn.utils import Dataset, compute_ap
from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np


def compute_map(dataset: Dataset, model: modellib.MaskRCNN, config: Config, iou_threshold: float = 0.5) -> float:
    aps = []

    for image_id in dataset.image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        results = model.detect([image], verbose=0)[0]

        ap, precisions, recalls, overlaps = compute_ap(
            gt_boxes=gt_bbox,
            gt_class_ids=gt_class_id,
            gt_masks=gt_mask,
            pred_boxes=results['rois'],
            pred_class_ids=results['class_ids'],
            pred_scores=results['scores'],
            pred_masks=results['masks'],
            iou_threshold=iou_threshold
        )

        aps.append(ap)

    return np.mean(aps)