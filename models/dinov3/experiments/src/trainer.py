from __future__ import annotations

import contextlib
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from .constants import CLASS_NAMES
from .evaluation import evaluate_model
from .losses import DetectionLoss
from .utils import append_jsonl, count_parameters, get_device, write_json



def create_optimizer(model, config):
    backbone_params = [param for param in model.backbone.parameters() if param.requires_grad]
    head_params = []
    for module in [
        model.neck,
        model.cls_tower,
        model.reg_tower,
        model.objectness_head,
        model.centerness_head,
        model.class_head,
        model.box_head,
        model.box_scale,
    ]:
        head_params.extend(param for param in module.parameters() if param.requires_grad)

    parameter_groups = []
    if backbone_params:
        parameter_groups.append({'params': backbone_params, 'lr': config.backbone_lr})
    if head_params:
        parameter_groups.append({'params': head_params, 'lr': config.head_lr})
    return AdamW(parameter_groups, weight_decay=config.weight_decay)



def save_checkpoint(model, output_path: str | Path, metrics, epoch: int, config) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'epoch': epoch,
            'class_names': CLASS_NAMES,
            'config': config.to_dict(),
        },
        output_path,
    )



def train_detection_model(model, train_loader, val_loader, config):
    device = get_device()
    model.to(device)
    criterion = DetectionLoss(
        box_weight=config.box_weight,
        giou_weight=config.giou_weight,
        obj_weight=config.obj_weight,
        cls_weight=config.cls_weight,
        positive_radius=config.positive_radius,
        centerness_weight=config.centerness_weight,
    )
    optimizer = create_optimizer(model, config)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available() and config.amp)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'metrics_summary.json'
    log_path = output_dir / 'training_log.jsonl'
    config_path = output_dir / 'resolved_config.json'
    write_json(config_path, config.to_dict())

    best_metrics = {'p': 0.0, 'r': 0.0, 'map50': 0.0, 'best_f1': 0.0, 'best_threshold': 0.0, 'per_class_ap50': {}}
    best_epoch = -1
    best_map50 = -1.0
    total_params, trainable_params = count_parameters(model)

    write_json(
        summary_path,
        {
            'status': 'running',
            'dataset_root': config.dataset_root,
            'img_size': config.img_size,
            'epoch': 0,
            'epochs': config.epochs,
            'best_epoch': best_epoch,
            'last_loss': None,
            'last_metrics': None,
            'best_metrics': best_metrics,
            'param': total_params,
            'parm': total_params,
            'trainable_param': trainable_params,
            'final_model_path': None,
        },
    )

    print(f'使用设备: {device}')
    print('开始训练优化版 DINOv3 检测模型...')

    for epoch in range(config.epochs):
        if epoch == config.warmup_epochs:
            model.unfreeze_last_blocks(config.unfreeze_blocks)
            optimizer = create_optimizer(model, config)
            scheduler = CosineAnnealingLR(optimizer, T_max=max(config.epochs - epoch, 1))
            print(f'已解冻最后 {config.unfreeze_blocks} 个 backbone blocks，开始联合微调。')

        model.train()
        epoch_loss = 0.0
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.epochs}')
        autocast_context = (
            torch.autocast(device_type='cuda', dtype=torch.float16)
            if torch.cuda.is_available() and config.amp
            else contextlib.nullcontext()
        )

        for images, targets in train_bar:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context:
                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())
            train_bar.set_postfix(
                {
                    'loss': f'{loss.item():.4f}',
                    'bbox': f"{loss_items['bbox']:.4f}",
                    'giou': f"{loss_items['giou']:.4f}",
                    'obj': f"{loss_items['obj']:.4f}",
                    'ctr': f"{loss_items['ctr']:.4f}",
                    'cls': f"{loss_items['cls']:.4f}",
                }
            )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, config, device)

        print(
            f"Epoch {epoch + 1}: "
            f"loss={avg_loss:.4f}, "
            f"p={val_metrics['p']:.4f}, "
            f"r={val_metrics['r']:.4f}, "
            f"map50={val_metrics['map50']:.4f}, "
            f"best_thr={val_metrics['best_threshold']:.4f}, "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        save_checkpoint(model, output_dir / 'last_visdrone_detection.pth', val_metrics, epoch + 1, config)
        if val_metrics['map50'] > best_map50:
            best_map50 = val_metrics['map50']
            best_metrics = val_metrics
            best_epoch = epoch + 1
            save_checkpoint(model, output_dir / 'best_visdrone_detection.pth', val_metrics, epoch + 1, config)

        epoch_record = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'metrics': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
        }
        append_jsonl(log_path, epoch_record)
        write_json(
            summary_path,
            {
                'status': 'running',
                'dataset_root': config.dataset_root,
                'img_size': config.img_size,
                'epoch': epoch + 1,
                'epochs': config.epochs,
                'best_epoch': best_epoch,
                'last_loss': avg_loss,
                'last_metrics': val_metrics,
                'best_metrics': best_metrics,
                'param': total_params,
                'parm': total_params,
                'trainable_param': trainable_params,
                'final_model_path': None,
            },
        )

    final_model_path = output_dir / 'final_visdrone_detection.pth'
    save_checkpoint(model, final_model_path, best_metrics, best_epoch, config)
    write_json(
        summary_path,
        {
            'status': 'completed',
            'dataset_root': config.dataset_root,
            'img_size': config.img_size,
            'epoch': config.epochs,
            'epochs': config.epochs,
            'best_epoch': best_epoch,
            'last_loss': avg_loss if config.epochs > 0 else None,
            'last_metrics': val_metrics if config.epochs > 0 else None,
            'best_metrics': best_metrics,
            'param': total_params,
            'parm': total_params,
            'trainable_param': trainable_params,
            'final_model_path': str(final_model_path),
        },
    )
    return model, best_metrics, best_epoch
