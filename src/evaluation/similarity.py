"""Análisis de similitud coseno: intra/inter clase, nearest neighbors."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity


def cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    """Calcula la matriz de similitud coseno (N, N)."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / (norms + 1e-8)
    return X_norm @ X_norm.T


def intra_inter_distributions(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retorna (intra_sims, inter_sims) — similitudes coseno intra-clase e inter-clase.
    Usa subsample si N > max_samples para eficiencia.
    """
    rng = np.random.default_rng(seed)
    if len(X) > max_samples:
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    sim = cosine_sim_matrix(X)
    N = len(y)
    intra, inter = [], []

    for i in range(N):
        for j in range(i + 1, N):
            if y[i] == y[j]:
                intra.append(sim[i, j])
            else:
                inter.append(sim[i, j])

    return np.array(intra), np.array(inter)


def plot_similarity_distributions(
    intra: np.ndarray,
    inter: np.ndarray,
    save_path: str | Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(intra, label="Intra-clase", color="steelblue", alpha=0.6, bins=50, stat="density", ax=ax)
    sns.histplot(inter, label="Inter-clase", color="salmon", alpha=0.6, bins=50, stat="density", ax=ax)
    ax.axvline(np.mean(intra), color="steelblue", linestyle="--", lw=1.5, label=f"Mean intra={np.mean(intra):.3f}")
    ax.axvline(np.mean(inter), color="salmon", linestyle="--", lw=1.5, label=f"Mean inter={np.mean(inter):.3f}")
    ax.set_xlabel("Similitud coseno")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribución de similitudes coseno: intra-clase vs inter-clase")
    ax.legend()
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def find_nearest_neighbors(
    X: np.ndarray,
    y: np.ndarray,
    query_indices: list[int],
    k: int = 5,
) -> list[dict]:
    """Encuentra los k vecinos más cercanos (por similitud coseno) para cada query."""
    sim = cosine_sim_matrix(X)
    results = []
    for qi in query_indices:
        sims = sim[qi].copy()
        sims[qi] = -1   # excluir el propio query
        top_k = np.argsort(sims)[::-1][:k]
        results.append({
            "query_idx": qi,
            "query_label": int(y[qi]),
            "neighbors": [
                {"idx": int(j), "label": int(y[j]), "similarity": float(sim[qi, j])}
                for j in top_k
            ],
        })
    return results


def plot_similarity_heatmap(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 100,
    save_path: str | Path | None = None,
    seed: int = 42,
) -> plt.Figure:
    """Heatmap de la matriz de similitud coseno para una muestra de imágenes."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(n_samples, len(X)), replace=False)
    idx = idx[np.argsort(y[idx])]   # ordenar por clase para visualización
    X_sub, y_sub = X[idx], y[idx]

    sim = cosine_sim_matrix(X_sub)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim, ax=ax, cmap="RdYlGn", vmin=-1, vmax=1, cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_title(f"Matriz de similitud coseno (n={len(idx)}, ordenado por clase)")
    ax.set_xlabel("Muestra")
    ax.set_ylabel("Muestra")

    # Separador entre clases
    n_pos = int(y_sub.sum())
    n_neg = len(y_sub) - n_pos
    ax.axhline(n_neg, color="black", lw=2)
    ax.axvline(n_neg, color="black", lw=2)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
