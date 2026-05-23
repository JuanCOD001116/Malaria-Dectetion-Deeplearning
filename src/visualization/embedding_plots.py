"""Plots específicos de embeddings: ROC curves, comparativas de reducción."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curves(
    models_results: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot de curvas ROC para todos los modelos en test.
    models_results: {model_name: {"y_true": ..., "y_prob": ...}}
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10.colors

    for i, (name, data) in enumerate(models_results.items()):
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{name} (AUC={roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curvas ROC — Comparación de modelos (Test)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_reduction_comparison(
    reeval_results: dict,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Comparación de accuracy: original vs PCA vs UMAP para top-2 modelos.
    reeval_results: {model_name: {dim_key: accuracy}}
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(reeval_results))
    model_names = list(reeval_results.keys())

    if not model_names:
        ax.set_title("Sin datos de reevaluación")
        return fig

    dim_keys = list(reeval_results[model_names[0]].keys())
    width = 0.8 / len(dim_keys)
    colors = ["steelblue", "salmon", "green"]

    for j, (dim_key, color) in enumerate(zip(dim_keys, colors)):
        vals = [reeval_results[m].get(dim_key, 0) for m in model_names]
        offset = (j - len(dim_keys) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=dim_key, color=color, alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (test)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Impacto de la reducción de dimensión en los top-2 modelos")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
