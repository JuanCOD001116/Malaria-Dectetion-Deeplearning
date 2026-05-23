"""Tests: lectura de imágenes, shapes, labels y split estratificado."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentations import get_eval_transform
from src.data.dataset import MalariaDataset, SupConPairDataset
from src.data.split import make_stratified_split


def test_malaria_dataset_len(tiny_csv):
    ds = MalariaDataset(tiny_csv, transform=get_eval_transform())
    assert len(ds) == 20


def test_malaria_dataset_item_shape(tiny_csv):
    ds = MalariaDataset(tiny_csv, transform=get_eval_transform(img_size=96))
    img, label = ds[0]
    assert img.shape == (3, 96, 96)
    assert label in (0, 1)


def test_malaria_dataset_labels(tiny_csv):
    ds = MalariaDataset(tiny_csv, transform=get_eval_transform())
    labels = ds.labels
    assert len(labels) == 20
    assert set(labels) == {0, 1}


def test_supcon_pair_dataset(tiny_csv):
    from src.data.augmentations import get_contrastive_transform
    ds = SupConPairDataset(tiny_csv, transform=get_contrastive_transform())
    v1, v2, label = ds[0]
    assert v1.shape == (3, 96, 96)
    assert v2.shape == (3, 96, 96)
    assert label in (0, 1)
    # Las dos vistas deben ser diferentes por las augmentations aleatorias
    # (con probabilidad >99% serán distintas)


def test_split_stratified(tiny_dataset, tmp_path):
    train_df, val_df, test_df = make_stratified_split(
        dataset_root=tiny_dataset,
        processed_dir=tmp_path / "splits",
        train_frac=0.70,
        val_frac=0.15,
        seed=42,
    )
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 20
    # Balanceo aproximado (±1 muestra) en cada split
    for df in (train_df, val_df, test_df):
        pos = df["label"].sum()
        neg = len(df) - pos
        assert abs(pos - neg) <= 1, f"Split desequilibrado: {pos} pos vs {neg} neg"


def test_split_idempotent(tiny_dataset, tmp_path):
    """Dos llamadas consecutivas deben retornar el mismo split."""
    kwargs = dict(dataset_root=tiny_dataset, processed_dir=tmp_path / "splits", seed=42)
    train1, val1, test1 = make_stratified_split(**kwargs)
    train2, val2, test2 = make_stratified_split(**kwargs)
    assert list(train1["path"]) == list(train2["path"])
