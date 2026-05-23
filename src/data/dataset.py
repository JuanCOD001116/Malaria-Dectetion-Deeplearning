"""Datasets PyTorch para malaria: evaluación y entrenamiento contrastivo."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class MalariaDataset(Dataset):
    """Dataset estándar (imagen, label) para extracción de embeddings y evaluación."""

    def __init__(self, csv_path: str | Path, transform: Callable | None = None) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(row["label"])

    @property
    def labels(self) -> list[int]:
        return self.df["label"].tolist()


class SupConPairDataset(Dataset):
    """
    Dataset para entrenamiento SupCon.
    Devuelve dos vistas augmentadas de la misma imagen: (view1, view2, label).
    El batch efectivo es 2N — estándar en SupCon/SimCLR.
    """

    def __init__(self, csv_path: str | Path, transform: Callable) -> None:
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, int(row["label"])
