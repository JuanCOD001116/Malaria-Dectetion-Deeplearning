"""
Loop de entrenamiento Supervised Contrastive Learning.
Soporta CPU y GPU (mixed precision automático en GPU).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.encoder import ContrastiveEncoder
from src.training.contrastive_loss import SupConLoss
from src.utils.io import save_checkpoint
from src.utils.logging import get_logger

logger = get_logger("train_contrastive")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info("GPU detectada: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("Entrenando en CPU (lento — considera usar Colab con GPU)")
    return torch.device("cpu")


def _build_optimizer(model: torch.nn.Module, cfg: dict) -> optim.Optimizer:
    opt_cfg = cfg.get("optimizer", {})
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 1e-3)
    wd = opt_cfg.get("weight_decay", 1e-4)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


def _build_scheduler(optimizer: optim.Optimizer, cfg: dict) -> Any:
    sch_cfg = cfg.get("scheduler", {})
    epochs = cfg.get("training", {}).get("epochs", 30)
    eta_min = sch_cfg.get("eta_min", 1e-5)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)


def train_one_epoch(
    model: ContrastiveEncoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: SupConLoss,
    device: torch.device,
    scaler,  # GradScaler o None
) -> float:
    model.train()
    total_loss = 0.0
    for v1, v2, labels in loader:
        # Concatenar las dos vistas para el batch SupCon
        views = torch.cat([v1, v2], dim=0).to(device)          # (2B, C, H, W)
        labels_rep = torch.cat([labels, labels], dim=0).to(device)  # (2B,)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                proj = model(views, return_embedding=False)     # (2B, 128) normalizado
                loss = loss_fn(proj, labels_rep)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            proj = model(views, return_embedding=False)
            loss = loss_fn(proj, labels_rep)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate_epoch(
    model: ContrastiveEncoder,
    loader: DataLoader,
    loss_fn: SupConLoss,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    for v1, v2, labels in loader:
        views = torch.cat([v1, v2], dim=0).to(device)
        labels_rep = torch.cat([labels, labels], dim=0).to(device)
        proj = model(views, return_embedding=False)
        loss = loss_fn(proj, labels_rep)
        total_loss += loss.item()
    return total_loss / len(loader)


def train(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str | Path = "artifacts/checkpoints",
) -> dict[str, list[float]]:
    """
    Entrena el encoder contrastivo y guarda checkpoints.
    Retorna historial de pérdidas {train: [...], val: [...]}.
    """
    device = _get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enc_cfg = cfg.get("encoder", {})
    model = ContrastiveEncoder(
        embedding_dim=enc_cfg.get("embedding_dim", 1024),
        proj_dim=enc_cfg.get("proj_dim", 128),
        pretrained=enc_cfg.get("pretrained", True),
    ).to(device)

    loss_fn = SupConLoss(temperature=cfg.get("loss", {}).get("temperature", 0.07))
    optimizer = _build_optimizer(model, cfg)
    scheduler = _build_scheduler(optimizer, cfg)

    # Mixed precision solo en GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    train_cfg = cfg.get("training", {})
    epochs = train_cfg.get("epochs", 30)
    patience = train_cfg.get("early_stopping_patience", 7)
    best_model_name = cfg.get("output", {}).get("best_model_name", "encoder_best.pt")
    last_model_name = cfg.get("output", {}).get("last_model_name", "encoder_last.pt")

    history = {"train": [], "val": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
        val_loss = evaluate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        lr = scheduler.get_last_lr()[0]

        logger.info(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | lr={lr:.6f}")

        # Checkpoint del mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_loss": val_loss, "config": cfg},
                output_dir / best_model_name,
            )
            logger.info(f"  ✓ Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping en epoch {epoch}")
                break

        # Checkpoint del último epoch siempre
        save_checkpoint(
            {"epoch": epoch, "model_state_dict": model.state_dict(),
             "history": history},
            output_dir / last_model_name,
        )

    logger.info(f"Entrenamiento completado. Mejor val_loss: {best_val_loss:.4f}")
    return history
