from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import apply_overrides, load_config
from src.constants import CLASS_NAMES
from src.dataset import create_dataloaders
from src.model import DINOv3DetectionModel
from src.trainer import train_detection_model
from src.utils import count_parameters, seed_everything



def parse_args():
    parser = argparse.ArgumentParser(description='200 epochs 版 DINOv3 VisDrone 检测训练入口')
    parser.add_argument('--config', default=str(PROJECT_ROOT / 'configs' / 'visdrone_dinov3_e200.json'))
    parser.add_argument('--dataset-root', dest='dataset_root')
    parser.add_argument('--output-dir', dest='output_dir')
    parser.add_argument('--backbone-name', dest='backbone_name')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--img-size', dest='img_size', type=int)
    parser.add_argument('--batch-size', dest='batch_size', type=int)
    parser.add_argument('--num-workers', dest='num_workers', type=int)
    parser.add_argument('--head-lr', dest='head_lr', type=float)
    parser.add_argument('--backbone-lr', dest='backbone_lr', type=float)
    parser.add_argument('--warmup-epochs', dest='warmup_epochs', type=int)
    parser.add_argument('--unfreeze-blocks', dest='unfreeze_blocks', type=int)
    parser.add_argument('--positive-radius', dest='positive_radius', type=int)
    parser.add_argument('--grad-clip-norm', dest='grad_clip_norm', type=float)
    parser.add_argument('--centerness-weight', dest='centerness_weight', type=float)
    parser.add_argument('--eval-conf-threshold', dest='eval_conf_threshold', type=float)
    parser.add_argument('--eval-pre-nms-topk', dest='eval_pre_nms_topk', type=int)
    parser.add_argument('--eval-max-predictions', dest='eval_max_predictions', type=int)
    parser.add_argument('--nms-threshold', dest='nms_threshold', type=float)
    parser.add_argument('--seed', type=int)
    return parser.parse_args()



def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, vars(args))
    seed_everything(config.seed)

    print('步骤 1/4: 创建 DINOv3 检测模型...')
    model = DINOv3DetectionModel(
        backbone_name=config.backbone_name,
        num_classes=len(CLASS_NAMES),
        img_size=config.img_size,
    )

    total_params, trainable_params = count_parameters(model)
    print(f'模型总参数量: {total_params}')
    print(f'当前可训练参数量: {trainable_params}')

    print('步骤 2/4: 加载 VisDrone 数据集...')
    train_loader, val_loader = create_dataloaders(
        dataset_root=config.dataset_root,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )
    print(f'训练集: {len(train_loader.dataset)} 样本')
    print(f'验证集: {len(val_loader.dataset)} 样本')

    print('步骤 3/4: 训练检测模型并评估...')
    _, best_metrics, best_epoch = train_detection_model(model, train_loader, val_loader, config)

    print('步骤 4/4: 输出最优指标...')
    print(f'best_epoch: {best_epoch}')
    print(f"p: {best_metrics['p']:.6f}")
    print(f"r: {best_metrics['r']:.6f}")
    print(f"map50: {best_metrics['map50']:.6f}")


if __name__ == '__main__':
    main()
