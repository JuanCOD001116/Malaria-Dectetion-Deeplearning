"""
Análisis de similitud coseno sobre embeddings de test.

Uso:
    python -m scripts.evaluate_similarity
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.similarity import (
    find_nearest_neighbors,
    intra_inter_distributions,
    plot_similarity_distributions,
    plot_similarity_heatmap,
)
from src.utils.io import load_config, load_embeddings, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("evaluate_similarity", log_dir="artifacts/logs")


def main() -> None:
    cfg = load_config("configs/data.yaml")
    set_global_seed(cfg["seed"])

    fig_dir = Path("artifacts/figures")
    out_dir = Path("artifacts/metrics")
    fig_dir.mkdir(parents=True, exist_ok=True)

    emb_dir = Path(cfg["embeddings_dir"])
    X_test, y_test = load_embeddings("test", emb_dir)
    logger.info(f"Test embeddings: {X_test.shape}")

    # Distribuciones intra/inter
    logger.info("Calculando similitudes coseno intra/inter-clase...")
    intra, inter = intra_inter_distributions(X_test, y_test, max_samples=500, seed=cfg["seed"])
    logger.info(f"  Intra-clase: mean={np.mean(intra):.4f} ± {np.std(intra):.4f}")
    logger.info(f"  Inter-clase: mean={np.mean(inter):.4f} ± {np.std(inter):.4f}")

    plot_similarity_distributions(intra, inter, save_path=fig_dir / "cosine_sim_distributions.png")

    # Heatmap
    plot_similarity_heatmap(X_test, y_test, n_samples=100,
                            save_path=fig_dir / "cosine_sim_heatmap.png",
                            seed=cfg["seed"])

    # Nearest neighbors: 5 queries por clase
    rng = np.random.default_rng(cfg["seed"])
    query_indices = []
    for label in [0, 1]:
        idx_pool = np.where(y_test == label)[0]
        chosen = rng.choice(idx_pool, min(5, len(idx_pool)), replace=False)
        query_indices.extend(chosen.tolist())

    nn_results = find_nearest_neighbors(X_test, y_test, query_indices, k=5)
    save_json(nn_results, out_dir / "nearest_neighbors.json")
    logger.info(f"Nearest neighbors guardados en {out_dir / 'nearest_neighbors.json'}")

    # Resumen estadístico
    summary = {
        "intra_class": {"mean": float(np.mean(intra)), "std": float(np.std(intra)),
                         "min": float(np.min(intra)), "max": float(np.max(intra))},
        "inter_class": {"mean": float(np.mean(inter)), "std": float(np.std(inter)),
                         "min": float(np.min(inter)), "max": float(np.max(inter))},
        "separability_gap": float(np.mean(intra) - np.mean(inter)),
    }
    save_json(summary, out_dir / "similarity_summary.json")
    logger.info(f"Separability gap: {summary['separability_gap']:.4f}")


if __name__ == "__main__":
    main()
