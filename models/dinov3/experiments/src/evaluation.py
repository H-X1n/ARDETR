from __future__ import annotations

import contextlib

import torch
from tqdm import tqdm

from .constants import CLASS_NAMES
from .utils import box_iou, cxcywh_to_xyxy, nms



def decode_predictions(outputs, conf_threshold=0.05, nms_threshold=0.5, max_predictions=300, pre_nms_topk=None):
    boxes_batch = outputs['boxes_xyxy'].detach().cpu().clamp(0.0, 1.0)
    objectness_batch = outputs['objectness'].detach().cpu().sigmoid()
    centerness_batch = outputs['centerness'].detach().cpu().sigmoid()
    class_probs_batch = outputs['class_logits'].detach().cpu().sigmoid()
    batch_predictions = []

    for boxes, objectness, centerness, class_probs in zip(boxes_batch, objectness_batch, centerness_batch, class_probs_batch):
        class_scores, class_ids = class_probs.max(dim=-1)
        scores = torch.sqrt((objectness * centerness).clamp(min=0.0)) * class_scores

        if pre_nms_topk is not None and scores.numel() > pre_nms_topk:
            topk_indices = torch.topk(scores, k=pre_nms_topk).indices
            boxes = boxes[topk_indices]
            scores = scores[topk_indices]
            class_ids = class_ids[topk_indices]

        keep_mask = scores >= conf_threshold
        boxes = boxes[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        if boxes.numel() == 0:
            batch_predictions.append({'boxes': torch.zeros((0, 4)), 'scores': torch.zeros((0,)), 'labels': torch.zeros((0,), dtype=torch.long)})
            continue

        final_indices = []
        for class_id in class_ids.unique():
            class_mask = class_ids == class_id
            selected = nms(boxes[class_mask], scores[class_mask], nms_threshold)
            final_indices.append(torch.where(class_mask)[0][selected])

        keep_indices = torch.cat(final_indices) if final_indices else torch.zeros((0,), dtype=torch.long)
        keep_indices = keep_indices[torch.argsort(scores[keep_indices], descending=True)[:max_predictions]]
        batch_predictions.append(
            {
                'boxes': boxes[keep_indices],
                'scores': scores[keep_indices],
                'labels': class_ids[keep_indices],
            }
        )

    return batch_predictions



def compute_ap(recalls, precisions):
    if not recalls:
        return 0.0
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 1, 0, -1):
        mpre[idx - 1] = max(mpre[idx - 1], mpre[idx])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap



def evaluate_predictions(predictions, targets, num_classes: int):
    gt_by_class = {class_id: {} for class_id in range(num_classes)}
    pred_by_class = {class_id: [] for class_id in range(num_classes)}

    for prediction, target in zip(predictions, targets):
        image_id = target['image_id']
        gt_boxes = cxcywh_to_xyxy(target['boxes']).cpu()
        gt_labels = target['labels'].cpu()

        for class_id in range(num_classes):
            class_gt_boxes = gt_boxes[gt_labels == class_id]
            if class_gt_boxes.numel() > 0:
                gt_by_class[class_id][image_id] = class_gt_boxes

        for pred_box, pred_score, pred_label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            pred_by_class[int(pred_label.item())].append(
                {'image_id': image_id, 'score': float(pred_score.item()), 'box': pred_box}
            )

    total_gt = sum(boxes.shape[0] for per_class in gt_by_class.values() for boxes in per_class.values())
    ap_values = []
    per_class_ap = {}
    threshold_records = []

    for class_id in range(num_classes):
        class_predictions = sorted(pred_by_class[class_id], key=lambda item: item['score'], reverse=True)
        class_gt = gt_by_class[class_id]
        num_gt = sum(boxes.shape[0] for boxes in class_gt.values())
        if num_gt == 0:
            continue

        matched = {image_id: torch.zeros(boxes.shape[0], dtype=torch.bool) for image_id, boxes in class_gt.items()}
        tp_flags = []
        fp_flags = []

        for prediction in class_predictions:
            image_id = prediction['image_id']
            pred_box = prediction['box'].unsqueeze(0)
            if image_id not in class_gt:
                tp_flags.append(0.0)
                fp_flags.append(1.0)
                continue

            gt_boxes = class_gt[image_id]
            ious = box_iou(pred_box, gt_boxes).squeeze(0)
            max_iou, max_index = ious.max(dim=0)
            if max_iou.item() >= 0.5 and not matched[image_id][max_index]:
                matched[image_id][max_index] = True
                tp_flags.append(1.0)
                fp_flags.append(0.0)
            else:
                tp_flags.append(0.0)
                fp_flags.append(1.0)

        cum_tp = 0.0
        cum_fp = 0.0
        recalls = []
        precisions = []
        for tp_flag, fp_flag in zip(tp_flags, fp_flags):
            cum_tp += tp_flag
            cum_fp += fp_flag
            recalls.append(cum_tp / max(num_gt, 1))
            precisions.append(cum_tp / max(cum_tp + cum_fp, 1e-6))

        ap = compute_ap(recalls, precisions)
        ap_values.append(ap)
        per_class_ap[CLASS_NAMES[class_id]] = ap

        for rank, prediction in enumerate(class_predictions):
            threshold_records.append({'score': prediction['score'], 'tp': tp_flags[rank], 'fp': fp_flags[rank]})

    threshold_records.sort(key=lambda item: item['score'], reverse=True)
    cum_tp = 0.0
    cum_fp = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_threshold = 0.0

    for record in threshold_records:
        cum_tp += record['tp']
        cum_fp += record['fp']
        precision = cum_tp / max(cum_tp + cum_fp, 1e-6)
        recall = cum_tp / max(total_gt, 1e-6)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-6)
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_threshold = record['score']

    map50 = sum(ap_values) / len(ap_values) if ap_values else 0.0
    return {
        'p': best_precision,
        'r': best_recall,
        'map50': map50,
        'best_f1': best_f1,
        'best_threshold': best_threshold,
        'per_class_ap50': per_class_ap,
    }



def evaluate_model(model, val_loader, config, device):
    model.eval()
    all_predictions = []
    all_targets = []
    autocast_context = (
        torch.autocast(device_type='cuda', dtype=torch.float16)
        if torch.cuda.is_available() and config.amp
        else contextlib.nullcontext()
    )

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='验证中', leave=False):
            images = images.to(device)
            with autocast_context:
                outputs = model(images)
            predictions = decode_predictions(
                outputs,
                conf_threshold=config.eval_conf_threshold,
                nms_threshold=config.nms_threshold,
                max_predictions=config.eval_max_predictions,
                pre_nms_topk=config.eval_pre_nms_topk,
            )
            all_predictions.extend(predictions)
            all_targets.extend(targets)

    return evaluate_predictions(all_predictions, all_targets, len(CLASS_NAMES))
