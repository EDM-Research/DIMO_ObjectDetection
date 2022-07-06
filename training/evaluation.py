from mrcnn.utils import Dataset, compute_ap, compute_matches
from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np


class ModelTest:
    def __init__(self, model_id: str, test_subset: str):
        self.model_id = model_id
        self.test_subset = test_subset
        self.metrics = {}


def compute_pandr(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5):
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold)

    precision = np.sum(pred_match > -1) / len(pred_match)
    recall = np.sum(pred_match > -1) / len(gt_match)

    return precision, recall


def compute_mean_pand(results: list, dataset: Dataset, config: Config, iou_threshold: float = 0.5) -> tuple:
    precisions = []
    recalls = []

    for result in results:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, result['image_id'])
        p, r = compute_pandr(
            gt_boxes=gt_bbox,
            gt_class_ids=gt_class_id,
            gt_masks=gt_mask,
            pred_boxes=result['rois'],
            pred_class_ids=result['class_ids'],
            pred_scores=result['scores'],
            pred_masks=result['masks'],
            iou_threshold=iou_threshold
        )

        precisions.append(p)
        recalls.append(r)

    return np.mean(precisions), np.mean(recalls)


def compute_map(results: list, dataset: Dataset, config: Config, iou_threshold: float = 0.5) -> float:
    aps = []

    for result in results:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config, result['image_id'])
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


def compute_coco_ap(results: list, dataset: Dataset, config: Config) -> float:
    thresholds = np.arange(start=0.50, stop=1.0, step=0.05)
    aps = [compute_map(results, dataset, config, threshold) for threshold in thresholds]

    return np.mean(aps)


