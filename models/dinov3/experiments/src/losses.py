from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import box_iou, cxcywh_to_xyxy, sigmoid_focal_loss


def compute_centerness_targets(ltrb_targets: torch.Tensor) -> torch.Tensor:
    left, top, right, bottom = ltrb_targets.unbind(dim=-1)
    horizontal = torch.minimum(left, right) / torch.maximum(left, right).clamp(min=1e-6)
    vertical = torch.minimum(top, bottom) / torch.maximum(top, bottom).clamp(min=1e-6)
    return torch.sqrt((horizontal * vertical).clamp(min=0.0))


def build_targets(targets, locations, grid_shape, device, positive_radius: int, num_classes: int):
    grid_h, grid_w = grid_shape
    num_locations = grid_h * grid_w
    batch_size = len(targets)
    locations = locations.to(device)
    location_x = locations[:, 0]
    location_y = locations[:, 1]

    box_targets_xyxy = torch.zeros((batch_size, num_locations, 4), device=device)
    box_targets_ltrb = torch.zeros((batch_size, num_locations, 4), device=device)
    cls_targets = torch.zeros((batch_size, num_locations, num_classes), device=device)
    obj_targets = torch.zeros((batch_size, num_locations), device=device)
    centerness_targets = torch.zeros((batch_size, num_locations), device=device)
    positive_mask = torch.zeros((batch_size, num_locations), dtype=torch.bool, device=device)
    assignment_area = torch.full((batch_size, num_locations), float('inf'), dtype=torch.float32, device=device)

    for batch_idx, target in enumerate(targets):
        boxes = target['boxes'].to(device)
        labels = target['labels'].to(device)
        if boxes.numel() == 0:
            continue

        gt_xyxy = cxcywh_to_xyxy(boxes)
        gt_areas = boxes[:, 2] * boxes[:, 3]
        order = torch.argsort(gt_areas, descending=False)
        for obj_idx in order.tolist():
            box = boxes[obj_idx]
            box_xyxy = gt_xyxy[obj_idx]
            x1, y1, x2, y2 = box_xyxy
            cx, cy, _, _ = box
            offsets = torch.stack(
                [
                    location_x - x1,
                    location_y - y1,
                    x2 - location_x,
                    y2 - location_y,
                ],
                dim=-1,
            )
            inside_box = offsets.min(dim=-1).values > 0
            center_radius_x = positive_radius / grid_w
            center_radius_y = positive_radius / grid_h
            inside_center = (
                (location_x >= cx - center_radius_x)
                & (location_x <= cx + center_radius_x)
                & (location_y >= cy - center_radius_y)
                & (location_y <= cy + center_radius_y)
            )
            candidate_mask = inside_box & inside_center
            if not candidate_mask.any():
                center_dist = (location_x - cx) ** 2 + (location_y - cy) ** 2
                if inside_box.any():
                    inside_indices = torch.where(inside_box)[0]
                    best_index = inside_indices[torch.argmin(center_dist[inside_indices])]
                else:
                    best_index = torch.argmin(center_dist)
                candidate_mask[best_index] = True

            update_mask = candidate_mask & (gt_areas[obj_idx] < assignment_area[batch_idx])
            if not update_mask.any():
                continue

            assigned_offsets = offsets[update_mask].clamp(min=1e-6)
            assignment_area[batch_idx, update_mask] = gt_areas[obj_idx]
            box_targets_xyxy[batch_idx, update_mask] = box_xyxy
            box_targets_ltrb[batch_idx, update_mask] = assigned_offsets
            cls_targets[batch_idx, update_mask] = 0.0
            cls_targets[batch_idx, update_mask, labels[obj_idx]] = 1.0
            obj_targets[batch_idx, update_mask] = 1.0
            centerness_targets[batch_idx, update_mask] = compute_centerness_targets(assigned_offsets)
            positive_mask[batch_idx, update_mask] = True

    return box_targets_xyxy, box_targets_ltrb, cls_targets, obj_targets, centerness_targets, positive_mask


class DetectionLoss(nn.Module):
    def __init__(
        self,
        box_weight: float = 5.0,
        giou_weight: float = 2.0,
        obj_weight: float = 1.0,
        cls_weight: float = 1.5,
        centerness_weight: float = 0.5,
        positive_radius: int = 1,
    ):
        super().__init__()
        self.box_weight = box_weight
        self.giou_weight = giou_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight
        self.centerness_weight = centerness_weight
        self.positive_radius = positive_radius

    def forward(self, outputs, targets):
        box_targets_xyxy, box_targets_ltrb, cls_targets, obj_targets, centerness_targets, positive_mask = build_targets(
            targets,
            outputs['locations'],
            outputs['grid_shape'],
            outputs['boxes_xyxy'].device,
            self.positive_radius,
            outputs['class_logits'].shape[-1],
        )

        loss_obj = sigmoid_focal_loss(outputs['objectness'], obj_targets)
        loss_cls = sigmoid_focal_loss(outputs['class_logits'], cls_targets)
        if positive_mask.any():
            pred_boxes = outputs['boxes_xyxy'][positive_mask]
            gt_boxes = box_targets_xyxy[positive_mask]
            pred_ltrb = outputs['box_deltas'][positive_mask]
            gt_ltrb = box_targets_ltrb[positive_mask]
            positive_weights = centerness_targets[positive_mask].detach().clamp(min=0.1)
            l1_per_box = F.smooth_l1_loss(pred_ltrb, gt_ltrb, beta=0.1, reduction='none').mean(dim=-1)
            loss_l1 = (l1_per_box * positive_weights).sum() / positive_weights.sum()

            iou = box_iou(pred_boxes, gt_boxes).diag()
            lt = torch.minimum(pred_boxes[:, :2], gt_boxes[:, :2])
            rb = torch.maximum(pred_boxes[:, 2:], gt_boxes[:, 2:])
            wh = (rb - lt).clamp(min=0)
            convex_area = wh[:, 0] * wh[:, 1]
            area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=0)
            area2 = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=0) * (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=0)
            inter_lt = torch.maximum(pred_boxes[:, :2], gt_boxes[:, :2])
            inter_rb = torch.minimum(pred_boxes[:, 2:], gt_boxes[:, 2:])
            inter_wh = (inter_rb - inter_lt).clamp(min=0)
            inter_area = inter_wh[:, 0] * inter_wh[:, 1]
            union = area1 + area2 - inter_area
            giou = iou - (convex_area - union) / convex_area.clamp(min=1e-6)
            loss_giou = ((1.0 - giou) * positive_weights).sum() / positive_weights.sum()
            centerness_loss = F.binary_cross_entropy_with_logits(
                outputs['centerness'][positive_mask],
                centerness_targets[positive_mask],
                reduction='none',
            )
            loss_centerness = (centerness_loss * positive_weights).sum() / positive_weights.sum()
        else:
            loss_l1 = outputs['boxes_xyxy'].sum() * 0.0
            loss_giou = outputs['boxes_xyxy'].sum() * 0.0
            loss_cls = outputs['class_logits'].sum() * 0.0
            loss_centerness = outputs['centerness'].sum() * 0.0

        total_loss = (
            self.obj_weight * loss_obj
            + self.centerness_weight * loss_centerness
            + self.cls_weight * loss_cls
            + self.box_weight * loss_l1
            + self.giou_weight * loss_giou
        )
        return total_loss, {
            'bbox': float(loss_l1.item()),
            'giou': float(loss_giou.item()),
            'obj': float(loss_obj.item()),
            'ctr': float(loss_centerness.item()),
            'cls': float(loss_cls.item()),
        }
