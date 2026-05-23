"""
Extrae embeddings de 1024 dims para train/val/test usando el encoder entrenado.

Uso:
    python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt

Salida:
    data/embeddings/{train,val,test}_X.npy  (shape: N x 1024)
    data/embeddings/{train,val,test}_y.npy  (shape: N,)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.augmentations import get_eval_transform
from src.data.dataset import MalariaDataset
from src.models.encoder import ContrastiveEncoder
from src.utils.io import load_checkpoint, load_config, save_embeddings
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("extract_embeddings")


def extract(
    model: ContrastiveEncoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_embeddings, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="  Extrayendo"):
            imgs = imgs.to(device)
            emb = model(imgs, return_embedding=True).cpu().numpy()
            all_embeddings.append(emb)
            all_labels.append(labels.numpy())

    X = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)
    assert X.shape[1] == 1024, f"Embedding dim erróneo: {X.shape[1]} != 1024"
    return X, y


def main(checkpoint_path: str, cfg_path: str = "configs/contrastive.yaml") -> None:
    cfg = load_config(cfg_path)
    data_cfg = load_config("configs/data.yaml")
    set_global_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo: {device}")

    # Cargar modelo
    enc_cfg = cfg.get("encoder", {})
    model = ContrastiveEncoder(
        embedding_dim=enc_cfg.get("embedding_dim", 1024),
        proj_dim=enc_cfg.get("proj_dim", 128),
        pretrained=False,   # pesos vienen del checkpoint
    ).to(device)

    ckpt = load_checkpoint(checkpoint_path, device=str(device))
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Checkpoint cargado: {checkpoint_path}")
    logger.info(f"  Epoch: {ckpt.get('epoch', '?')} | val_loss: {ckpt.get('val_loss', '?'):.4f}")

    transform = get_eval_transform(img_size=data_cfg.get("img_size", 96))
    processed_dir = Path(data_cfg["processed_dir"])
    embeddings_dir = Path(data_cfg["embeddings_dir"])
    batch_size = 128

    for split in ["train", "val", "test"]:
        csv_path = processed_dir / f"{split}.csv"
        assert csv_path.exists(), f"No existe {csv_path} — ejecuta prepare_data primero"

        ds = MalariaDataset(csv_path, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

        logger.info(f"Extrayendo embeddings: {split} ({len(ds)} imágenes)")
        X, y = extract(model, loader, device)
        save_embeddings(X, y, split, embeddings_dir)
        logger.info(f"  {split}: shape={X.shape}, positivos={y.sum()}/{len(y)}")

    logger.info(f"Embeddings guardados en {embeddings_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/encoder_best.pt")
    parser.add_argument("--config", default="configs/contrastive.yaml")
    args = parser.parse_args()
    main(args.checkpoint, args.config)
