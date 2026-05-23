"""
Entrena el encoder contrastivo SupCon.

Uso local (CPU lento):
    python -m scripts.train_contrastive --config configs/contrastive.yaml

Uso en Colab (GPU):
    # Ver notebook 02_contrastive_training.ipynb

El batch_size se ajusta automáticamente según el dispositivo disponible.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from src.data.augmentations import get_contrastive_transform
from src.data.dataset import SupConPairDataset
from src.training.train_contrastive import train
from src.utils.io import load_config, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("scripts.train_contrastive", log_dir="artifacts/logs")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    data_cfg = load_config("configs/data.yaml")

    set_global_seed(cfg["seed"])

    processed_dir = Path(data_cfg["processed_dir"])
    assert (processed_dir / "train.csv").exists(), (
        "Ejecuta primero: python -m scripts.prepare_data --config configs/data.yaml"
    )

    # Batch size según dispositivo
    use_gpu = torch.cuda.is_available()
    train_cfg = cfg.get("training", {})
    batch_size = train_cfg.get("batch_size_gpu", 256) if use_gpu else train_cfg.get("batch_size_cpu", 32)
    num_workers = train_cfg.get("num_workers", 4) if use_gpu else 0  # 0 en CPU Windows
    logger.info(f"batch_size={batch_size} | num_workers={num_workers}")

    aug_cfg = cfg.get("augmentations", {})
    transform = get_contrastive_transform(
        img_size=aug_cfg.get("img_size", 96),
        crop_scale=(aug_cfg.get("crop_scale_min", 0.7), aug_cfg.get("crop_scale_max", 1.0)),
        color_jitter_strength=aug_cfg.get("color_jitter", {}).get("brightness", 0.2),
        blur_prob=aug_cfg.get("blur_prob", 0.3),
    )

    train_ds = SupConPairDataset(processed_dir / "train.csv", transform=transform)
    val_ds = SupConPairDataset(processed_dir / "val.csv", transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu,
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    output_dir = cfg.get("output", {}).get("checkpoint_dir", "artifacts/checkpoints")
    history = train(cfg=cfg, train_loader=train_loader, val_loader=val_loader, output_dir=output_dir)

    # Guardar historial de pérdidas
    save_json(history, "artifacts/logs/contrastive_history.json")
    logger.info("Historial guardado en artifacts/logs/contrastive_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/contrastive.yaml")
    args = parser.parse_args()
    main(args.config)
