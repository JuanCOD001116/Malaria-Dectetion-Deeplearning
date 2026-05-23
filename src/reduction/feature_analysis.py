"""Análisis individual de features: varianza, correlación con label, ranking."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pointbiserialr


def feature_variance(X: np.ndarray) -> np.ndarray:
    return np.var(X, axis=0)


def feature_discriminability(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Correlación point-biserial de cada feature con la label binaria."""
    n_features = X.shape[1]
    corrs = np.zeros(n_features)
    for i in range(n_features):
        corr, _ = pointbiserialr(y, X[:, i])
        corrs[i] = abs(corr)   # valor absoluto — interesa magnitud
    return corrs


def rank_features(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Combina varianza y discriminabilidad para ranking de features.
    Retorna dict con arrays de varianza, correlación y ranking combinado.
    """
    var = feature_variance(X)
    corr = feature_discriminability(X, y)
    # Ranking combinado: promedio de percentil de varianza y correlación
    var_rank = np.argsort(np.argsort(var))
    corr_rank = np.argsort(np.argsort(corr))
    combined = (var_rank + corr_rank) / 2
    return {"variance": var, "discriminability": corr, "combined_rank": combined}


def plot_top_features(
    ranking: dict,
    top_k: int = 50,
    save_path: str | Path | None = None,
) -> plt.Figure:
    corr = ranking["discriminability"]
    top_idx = np.argsort(corr)[::-1][:top_k]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top-k por discriminabilidad
    axes[0].bar(range(top_k), corr[top_idx], color="steelblue", alpha=0.8)
    axes[0].set_xlabel(f"Top-{top_k} features (índice ordenado)")
    axes[0].set_ylabel("|Correlación point-biserial|")
    axes[0].set_title(f"Top-{top_k} features más discriminativas")
    axes[0].set_xticks([])

    # Distribución de todas las discriminabilidades
    axes[1].hist(corr, bins=50, color="salmon", alpha=0.8, edgecolor="black")
    axes[1].set_xlabel("|Correlación point-biserial|")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title("Distribución de discriminabilidad (1024 features)")
    axes[1].axvline(np.mean(corr), color="darkred", linestyle="--", label=f"Media={np.mean(corr):.3f}")
    axes[1].legend()

    fig.suptitle("Análisis de features del embedding contrastivo", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
