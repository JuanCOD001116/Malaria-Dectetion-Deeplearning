"""UMAP: visualización 2D y reducción para clasificación."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def fit_umap(
    X_train: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    low_memory: bool = True,
    seed: int = 42,
):
    """Ajusta UMAP y retorna el reducer entrenado."""
    import umap

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        low_memory=low_memory,
        random_state=seed,
    )
    reducer.fit(X_train)
    return reducer


def transform_umap(reducer, X: np.ndarray) -> np.ndarray:
    return reducer.transform(X)


def plot_umap_2d(
    X_2d: np.ndarray,
    y: np.ndarray,
    title: str = "UMAP 2D — Embeddings contrastivos",
    class_names: list[str] | None = None,
    save_path: str | Path | None = None,
    alpha: float = 0.4,
    s: float = 5.0,
) -> plt.Figure:
    """Plot UMAP 2D coloreado por clase."""
    if class_names is None:
        class_names = ["Uninfected", "Parasitized"]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = ["royalblue", "tomato"]

    for label, name, color in zip([0, 1], class_names, colors):
        mask = y == label
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name,
                   alpha=alpha, s=s, edgecolors="none")

    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
