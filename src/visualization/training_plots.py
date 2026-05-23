"""Plots de curvas de entrenamiento del encoder contrastivo."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Curvas de pérdida contrastiva train/val por epoch."""
    train_losses = history.get("train", [])
    val_losses = history.get("val", [])
    epochs = range(1, len(train_losses) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, label="Train loss", color="steelblue", lw=2)
    ax.plot(epochs, val_losses, label="Val loss", color="tomato", lw=2, linestyle="--")

    best_epoch = val_losses.index(min(val_losses)) + 1
    ax.axvline(best_epoch, color="green", linestyle=":", lw=1.5, label=f"Best epoch={best_epoch}")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("SupCon Loss")
    ax.set_title("Curvas de entrenamiento — Supervised Contrastive Learning")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_metrics_comparison(
    summary: dict,
    metric: str = "test_f1_macro",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Barplot comparando una métrica entre todos los modelos clásicos."""
    models = list(summary.keys())
    values = [summary[m].get(metric, 0) for m in models]

    sorted_pairs = sorted(zip(values, models), reverse=True)
    values, models = zip(*sorted_pairs)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(models, values, color="steelblue", alpha=0.8)
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title(f"Comparación de modelos — {metric}")
    ax.set_xlim(0, 1.05)

    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
