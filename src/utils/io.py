"""I/O helpers: configs YAML, checkpoints PyTorch, embeddings NumPy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)


def save_json(data: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_embeddings(X: np.ndarray, y: np.ndarray, split: str, out_dir: str | Path) -> None:
    """Guarda embeddings y etiquetas en formato .npy."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_X.npy", X)
    np.save(out_dir / f"{split}_y.npy", y)
    assert X.shape[1] == 1024, f"Embeddings deben tener 1024 dims, got {X.shape[1]}"


def load_embeddings(split: str, embeddings_dir: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Carga embeddings y etiquetas desde .npy."""
    d = Path(embeddings_dir)
    X = np.load(d / f"{split}_X.npy")
    y = np.load(d / f"{split}_y.npy")
    return X, y


def save_checkpoint(state: dict, path: str | Path) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    import torch
    return torch.load(path, map_location=device, weights_only=False)
