"""Fixtures compartidas: crea un mini-dataset sintético en tmp_path."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path):
    """Crea 20 imágenes PNG sintéticas (10 Parasitized + 10 Uninfected)."""
    for cls, label in [("Parasitized", 1), ("Uninfected", 0)]:
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(10):
            arr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.png")

    return tmp_path


@pytest.fixture
def tiny_csv(tmp_path, tiny_dataset):
    """Genera un CSV de split sobre el mini-dataset."""
    records = []
    for cls, label in [("Parasitized", 1), ("Uninfected", 0)]:
        for img in sorted((tiny_dataset / cls).glob("*.png")):
            records.append({"path": str(img), "label": label, "class_name": cls})
    df = pd.DataFrame(records)
    csv_path = tmp_path / "train.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tiny_embeddings():
    """Embeddings sintéticos de 1024 dimensiones."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 1024)).astype(np.float32)
    y = np.array([1] * 10 + [0] * 10)
    return X, y
