"""Transforms para evaluación y para entrenamiento contrastivo."""
from __future__ import annotations

import torchvision.transforms as T


# Normalización estándar ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_eval_transform(img_size: int = 96) -> T.Compose:
    """Transform determinístico para extracción de embeddings y evaluación."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_contrastive_transform(
    img_size: int = 96,
    crop_scale: tuple[float, float] = (0.7, 1.0),
    color_jitter_strength: float = 0.2,
    blur_prob: float = 0.3,
) -> T.Compose:
    """
    Augmentations médicas suaves para SupCon training.
    Cada llamada produce una vista diferente de la misma imagen.
    Sin Cutout ni GridMask agresivos porque pueden borrar el parásito.
    """
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=crop_scale),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),   # celdas sin orientación canónica
        T.RandomRotation(degrees=15),
        T.ColorJitter(
            brightness=color_jitter_strength,
            contrast=color_jitter_strength,
            saturation=color_jitter_strength,
            hue=0.0,   # sin cambio de matiz — preserva color del parásito
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=blur_prob),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
