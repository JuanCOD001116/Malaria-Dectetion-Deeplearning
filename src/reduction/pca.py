"""PCA sobre embeddings: análisis de varianza explicada y reducción."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def fit_pca(
    X_train: np.ndarray,
    variance_threshold: float = 0.95,
    n_components_fixed: int | None = None,
    whiten: bool = False,
    seed: int = 42,
) -> tuple[PCA, int]:
    """
    Ajusta PCA sobre X_train.
    Si n_components_fixed es None, selecciona automáticamente los
    componentes que explican variance_threshold de la varianza.

    Returns:
        (pca_model, n_components_selected)
    """
    # Primera pasada: ajustar con todos los componentes para ver varianza acumulada
    pca_full = PCA(whiten=whiten, random_state=seed)
    pca_full.fit(X_train)

    if n_components_fixed is not None:
        n_comp = n_components_fixed
    else:
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cumvar, variance_threshold) + 1)

    pca = PCA(n_components=n_comp, whiten=whiten, random_state=seed)
    pca.fit(X_train)
    return pca, n_comp


def transform_pca(pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(X)


def plot_scree(
    pca: PCA,
    n_display: int = 50,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Scree plot: varianza explicada por componente y varianza acumulada."""
    evr = pca.explained_variance_ratio_
    n = min(n_display, len(evr))
    cumvar = np.cumsum(evr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("PCA — Análisis de varianza explicada", fontweight="bold")

    ax1.bar(range(1, n + 1), evr[:n] * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Componente principal")
    ax1.set_ylabel("Varianza explicada (%)")
    ax1.set_title(f"Top-{n} componentes")

    ax2.plot(range(1, len(cumvar) + 1), cumvar * 100, color="darkblue", lw=2)
    ax2.axhline(95, color="red", linestyle="--", label="95%")
    ax2.axhline(99, color="orange", linestyle="--", label="99%")
    ax2.set_xlabel("Número de componentes")
    ax2.set_ylabel("Varianza acumulada (%)")
    ax2.set_title("Varianza acumulada")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
