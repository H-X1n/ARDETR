from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=padding,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialPyramidContext(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 384, out_channels: int = 256):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, hidden_channels, kernel_size=1, padding=0)
        self.pool_5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.pool_9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.pool_13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.fuse = ConvBNAct(hidden_channels * 4, out_channels, kernel_size=1, padding=0)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(out_channels, dilation=1),
            DepthwiseSeparableConv(out_channels, dilation=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reduce(x)
        x = torch.cat([x, self.pool_5(x), self.pool_9(x), self.pool_13(x)], dim=1)
        x = self.fuse(x)
        return self.refine(x)


class DetectionTower(nn.Module):
    def __init__(self, channels: int, depth: int = 3):
        super().__init__()
        dilations = [1, 2, 1][:depth]
        self.layers = nn.Sequential(*[DepthwiseSeparableConv(channels, dilation=dilation) for dilation in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Scale(nn.Module):
    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class DINOv3DetectionModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, img_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            dummy_features = self._extract_feature_map(dummy_input)
            feature_dim = dummy_features.shape[1]
            self.grid_size = dummy_features.shape[-1]

        self.neck = SpatialPyramidContext(feature_dim, hidden_channels=384, out_channels=256)
        self.cls_tower = DetectionTower(256, depth=3)
        self.reg_tower = DetectionTower(256, depth=3)
        self.objectness_head = nn.Conv2d(256, 1, kernel_size=1)
        self.centerness_head = nn.Conv2d(256, 1, kernel_size=1)
        self.class_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.box_head = nn.Conv2d(256, 4, kernel_size=1)
        self.box_scale = Scale(1.0)
        self._init_detection_head()
        self.freeze_backbone()

    def _init_detection_head(self) -> None:
        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.objectness_head.bias, bias_value)
        nn.init.constant_(self.centerness_head.bias, 0.0)
        nn.init.constant_(self.class_head.bias, bias_value)
        nn.init.zeros_(self.box_head.bias)

    def freeze_backbone(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_blocks(self, block_count: int) -> None:
        self.freeze_backbone()
        blocks = getattr(self.backbone, 'blocks', None)
        if blocks is None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            return

        for block in blocks[-block_count:]:
            for param in block.parameters():
                param.requires_grad = True

        norm = getattr(self.backbone, 'norm', None)
        if norm is not None:
            for param in norm.parameters():
                param.requires_grad = True

    def _extract_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        if isinstance(features, dict):
            if 'x_norm_patchtokens' in features:
                features = features['x_norm_patchtokens']
            else:
                tensor_values = [value for value in features.values() if torch.is_tensor(value)]
                if not tensor_values:
                    raise RuntimeError('backbone.forward_features() 没有返回可用特征')
                features = tensor_values[-1]

        if features.ndim == 4:
            return features
        if features.ndim != 3:
            raise RuntimeError(f'不支持的特征维度: {features.shape}')

        patch_embed = getattr(self.backbone, 'patch_embed', None)
        patch_size = getattr(patch_embed, 'patch_size', 16)
        if isinstance(patch_size, tuple):
            patch_h, patch_w = patch_size
        else:
            patch_h = patch_w = patch_size

        expected_grid_h = x.shape[-2] // patch_h
        expected_grid_w = x.shape[-1] // patch_w
        expected_tokens = expected_grid_h * expected_grid_w
        token_count = features.shape[1]

        if token_count == expected_tokens:
            patch_tokens = features
        elif token_count > expected_tokens:
            prefix_tokens = token_count - expected_tokens
            patch_tokens = features[:, prefix_tokens:, :]
        else:
            square_grid = int(math.sqrt(token_count))
            if square_grid * square_grid != token_count:
                raise RuntimeError(
                    '无法将 token 序列重排为 2D 特征图: '
                    f'token_count={token_count}, expected_tokens={expected_tokens}'
                )
            expected_grid_h = expected_grid_w = square_grid
            patch_tokens = features

        return patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0],
            patch_tokens.shape[2],
            expected_grid_h,
            expected_grid_w,
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | tuple[int, int]]:
        feature_map = self._extract_feature_map(x)
        neck_features = self.neck(feature_map)
        cls_features = self.cls_tower(neck_features)
        reg_features = self.reg_tower(neck_features)
        box_raw = self.box_scale(self.box_head(reg_features))
        obj_logits = self.objectness_head(reg_features)
        centerness_logits = self.centerness_head(reg_features)
        cls_logits = self.class_head(cls_features)

        batch_size, _, grid_h, grid_w = box_raw.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=box_raw.device),
            torch.arange(grid_w, device=box_raw.device),
            indexing='ij',
        )

        center_x = (grid_x + 0.5) / grid_w
        center_y = (grid_y + 0.5) / grid_h
        left = F.softplus(box_raw[:, 0]) / grid_w
        top = F.softplus(box_raw[:, 1]) / grid_h
        right = F.softplus(box_raw[:, 2]) / grid_w
        bottom = F.softplus(box_raw[:, 3]) / grid_h
        pred_boxes_xyxy = torch.stack(
            [
                center_x.unsqueeze(0) - left,
                center_y.unsqueeze(0) - top,
                center_x.unsqueeze(0) + right,
                center_y.unsqueeze(0) + bottom,
            ],
            dim=-1,
        ).view(batch_size, grid_h * grid_w, 4)
        box_deltas = torch.stack([left, top, right, bottom], dim=-1).view(batch_size, grid_h * grid_w, 4)
        locations = torch.stack([center_x, center_y], dim=-1).view(grid_h * grid_w, 2)

        return {
            'boxes_xyxy': pred_boxes_xyxy,
            'box_deltas': box_deltas,
            'objectness': obj_logits.flatten(1),
            'centerness': centerness_logits.flatten(1),
            'class_logits': cls_logits.flatten(2).transpose(1, 2),
            'grid_shape': (grid_h, grid_w),
            'locations': locations,
        }
