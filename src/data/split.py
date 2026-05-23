"""Genera y carga splits estratificados fijos train/val/test."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def make_stratified_split(
    dataset_root: str | Path,
    processed_dir: str | Path,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    class_map: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Escanea dataset_root/{class_name}/*.png y genera splits estratificados.
    Idempotente: si los CSVs ya existen, los carga sin recalcular.

    Returns:
        (train_df, val_df, test_df) con columnas ['path', 'label', 'class_name']
    """
    processed_dir = Path(processed_dir)
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"
    test_csv = processed_dir / "test.csv"

    if train_csv.exists() and val_csv.exists() and test_csv.exists():
        return pd.read_csv(train_csv), pd.read_csv(val_csv), pd.read_csv(test_csv)

    processed_dir.mkdir(parents=True, exist_ok=True)

    if class_map is None:
        class_map = {"Parasitized": 1, "Uninfected": 0}

    records = []
    dataset_root = Path(dataset_root)
    for class_name, label in class_map.items():
        class_dir = dataset_root / class_name
        for img_path in sorted(class_dir.glob("*.png")):
            records.append({"path": str(img_path), "label": label, "class_name": class_name})
        # también buscar .jpg/.jpeg
        for img_path in sorted(class_dir.glob("*.jpg")):
            records.append({"path": str(img_path), "label": label, "class_name": class_name})

    df = pd.DataFrame(records)
    assert len(df) > 0, f"No se encontraron imágenes en {dataset_root}"

    # test_frac relativo al total
    test_frac = 1.0 - train_frac - val_frac

    train_df, temp_df = train_test_split(
        df, test_size=(val_frac + test_frac), stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(test_frac / (val_frac + test_frac)), stratify=temp_df["label"], random_state=seed
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    return train_df, val_df, test_df
