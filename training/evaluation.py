from mrcnn.utils import Dataset, compute_ap, compute_matches
from mrcnn.config import Config
import mrcnn.model as modellib
import numpy as np
import os


def get_file_for_epoch(model_dir: str, epoch: int = None) -> str:
    def get_epoch_no(file_name: str) -> int:
        return int(file_name.split('.')[0].split('_')[-1])

    last_epoch = 0
    last_epoch_file = ""

    for file in os.listdir(model_dir):
        if file.endswith('.h5'):
            current_epoch = get_epoch_no(file) - 1
            if current_epoch > last_epoch:
                last_epoch = current_epoch
                last_epoch_file = file
            if current_epoch == epoch:
                return file

    return last_epoch_file


def load_model(model_id: str, config: Config, epoch: int = None, mode: str = "inference") -> modellib.MaskRCNN:
    assert mode in ["training", "inference"], f"Mode can only be training or inference, not {mode}"
    model = modellib.MaskRCNN(mode=mode, config=config, model_dir=f"models")
    model_file = get_file_for_epoch(f"models/{model_id}", epoch)
    model_path = f"models/{model_id}/{model_file}"
    print(f"Loading model from {model_path}")
    model.load_weights(model_path, by_name=True)
    return model


def get_detections_dataset(dataset: Dataset, model: modellib.MaskRCNN, config: Config) -> list:
    results = []

    for i, image_id in enumerate(dataset.image_ids):
        print(f"Testing image {i}/{len(dataset.image_ids)}", end='\r')
        image, *_ = modellib.load_image_gt(dataset, config, image_id)
        result = model.detect([image], verbose=0)[0]
        result['image_id'] = image_id
        results.append(result)

    return results


def get_detections_images(images: list, model: modellib.MaskRCNN) -> list:
    results = []

    for i, image in enumerate(images):
        print(f"Testing image {i}/{len(images)}", end='\r')
        result = model.detect([image], verbose=0)[0]
        result['image_id'] = i
        results.append(result)

    return results


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


