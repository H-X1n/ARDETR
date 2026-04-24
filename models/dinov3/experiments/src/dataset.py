from __future__ import annotations

from pathlib import Path

import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class VisDroneDetectionDataset(Dataset):
    def __init__(self, root_dir: str, split: str, img_size: int, augment: bool = False):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / split / 'images'
        self.labels_dir = self.root_dir / split / 'labels'
        self.augment = augment

        if not self.images_dir.exists() or not self.labels_dir.exists():
            raise FileNotFoundError(f'数据目录不存在: {self.images_dir} 或 {self.labels_dir}')

        self.image_paths = sorted(
            [
                path
                for path in self.images_dir.iterdir()
                if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}
            ]
        )
        if not self.image_paths:
            raise RuntimeError(f'未在 {self.images_dir} 找到图像')

        self.to_tensor = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((img_size, img_size)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = self.labels_dir / f'{image_path.stem}.txt'
        image = Image.open(image_path).convert('RGB')
        boxes, labels = self._read_label_file(label_path)
        image, boxes = self._apply_augmentations(image, boxes)
        image_tensor = self.to_tensor(image)
        return image_tensor, {'boxes': boxes, 'labels': labels, 'image_id': image_path.stem}

    def _apply_augmentations(self, image: Image.Image, boxes: torch.Tensor):
        if not self.augment:
            return image, boxes

        if torch.rand(1).item() < 0.5:
            image = torchvision.transforms.functional.hflip(image)
            if boxes.numel() > 0:
                boxes = boxes.clone()
                boxes[:, 0] = 1.0 - boxes[:, 0]

        if torch.rand(1).item() < 0.8:
            image = torchvision.transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.15,
                hue=0.05,
            )(image)

        return image, boxes

    def _read_label_file(self, label_path: Path):
        if not label_path.exists():
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)

        boxes = []
        labels = []
        with label_path.open('r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id = int(float(parts[0]))
                if class_id < 0 or class_id > 9:
                    continue
                cx, cy, w, h = map(float, parts[1:])
                if w <= 0 or h <= 0:
                    continue
                boxes.append([
                    min(max(cx, 0.0), 1.0),
                    min(max(cy, 0.0), 1.0),
                    min(max(w, 1e-4), 1.0),
                    min(max(h, 1e-4), 1.0),
                ])
                labels.append(class_id)

        if not boxes:
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.long)
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)



def collate_fn(batch):
    images = [sample[0] for sample in batch]
    targets = [sample[1] for sample in batch]
    return torch.stack(images), targets



def create_dataloaders(dataset_root: str, img_size: int, batch_size: int, num_workers: int):
    train_dataset = VisDroneDetectionDataset(
        dataset_root,
        'VisDrone2019-DET-train',
        img_size,
        augment=True,
    )
    val_dataset = VisDroneDetectionDataset(
        dataset_root,
        'VisDrone2019-DET-val',
        img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader
