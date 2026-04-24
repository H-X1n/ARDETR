from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    seed: int = 42
    dataset_root: str = '/mnt/storage/sports/dinov3/VisDrone/VisDrone_Dataset'
    backbone_name: str = 'vit_large_patch16_dinov3.lvd1689m'
    output_dir: str = 'visdrone_dinov3_detector_e200/outputs/run01'
    img_size: int = 640
    batch_size: int = 2
    num_workers: int = 4
    epochs: int = 200
    head_lr: float = 3e-4
    backbone_lr: float = 3e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 2
    unfreeze_blocks: int = 6
    positive_radius: int = 2
    grad_clip_norm: float = 1.0
    centerness_weight: float = 0.5
    box_weight: float = 5.0
    giou_weight: float = 2.0
    obj_weight: float = 1.0
    cls_weight: float = 1.5
    nms_threshold: float = 0.5
    eval_conf_threshold: float = 1e-4
    eval_pre_nms_topk: int = 1500
    eval_max_predictions: int = 1000
    amp: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_OVERRIDABLE_FIELDS = set(TrainConfig.__dataclass_fields__.keys())


def load_config(config_path: str | Path | None = None) -> TrainConfig:
    if config_path is None:
        return TrainConfig()

    path = Path(config_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    return TrainConfig(**data)


def apply_overrides(config: TrainConfig, overrides: dict[str, Any]) -> TrainConfig:
    for key, value in overrides.items():
        if key not in _OVERRIDABLE_FIELDS or value is None:
            continue
        setattr(config, key, value)
    return config
