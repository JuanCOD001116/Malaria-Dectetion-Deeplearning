"""Tests: ranges y determinismo de augmentations."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentations import get_contrastive_transform, get_eval_transform


def _random_pil(h=96, w=96):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_eval_transform_shape():
    img = _random_pil(128, 120)
    t = get_eval_transform(img_size=96)
    out = t(img)
    assert out.shape == (3, 96, 96)


def test_eval_transform_normalized():
    img = _random_pil()
    t = get_eval_transform(img_size=96)
    out = t(img)
    # Después de normalización ImageNet los valores deben ser aprox en [-3, 3]
    assert out.min().item() > -5.0
    assert out.max().item() < 5.0


def test_contrastive_transform_shape():
    img = _random_pil(128, 128)
    t = get_contrastive_transform(img_size=96)
    v1 = t(img)
    v2 = t(img)
    assert v1.shape == (3, 96, 96)
    assert v2.shape == (3, 96, 96)


def test_eval_transform_deterministic():
    """El mismo input con eval_transform debe producir el mismo output."""
    img = _random_pil()
    t = get_eval_transform()
    out1 = t(img)
    out2 = t(img)
    assert torch.allclose(out1, out2)
