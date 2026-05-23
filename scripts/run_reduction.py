"""
PCA + UMAP sobre embeddings y reevaluación de los top-2 modelos.

Uso:
    python -m scripts.run_reduction --config configs/reduction.yaml
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.evaluation.metrics import evaluate_model
from src.reduction.feature_analysis import plot_top_features, rank_features
from src.reduction.pca import fit_pca, plot_scree, transform_pca
from src.reduction.umap_reducer import fit_umap, plot_umap_2d, transform_umap
from src.utils.io import load_config, load_embeddings, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("run_reduction", log_dir="artifacts/logs")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    emb_dir = Path(cfg["embeddings_dir"])
    out_dir = Path(cfg["output_dir"])
    fig_dir = Path(cfg["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Cargando embeddings...")
    X_train, y_train = load_embeddings("train", emb_dir)
    X_val, y_val = load_embeddings("val", emb_dir)
    X_test, y_test = load_embeddings("test", emb_dir)

    # ── Análisis de features ──────────────────────────────────────────────────
    logger.info("Análisis de features individuales...")
    ranking = rank_features(X_train, y_train)
    top_k = cfg.get("feature_analysis", {}).get("top_k", 50)
    plot_top_features(ranking, top_k=top_k, save_path=fig_dir / "feature_analysis.png")
    logger.info(f"  Discriminabilidad media: {np.mean(ranking['discriminability']):.4f}")

    # ── PCA ───────────────────────────────────────────────────────────────────
    logger.info("Ajustando PCA...")
    pca_cfg = cfg.get("pca", {})
    pca, n_comp = fit_pca(
        X_train,
        variance_threshold=pca_cfg.get("variance_threshold", 0.95),
        n_components_fixed=pca_cfg.get("n_components_fixed"),
        seed=cfg["seed"],
    )
    logger.info(f"  PCA: {n_comp} componentes explican ≥{pca_cfg.get('variance_threshold', 0.95)*100:.0f}% varianza")

    plot_scree(pca, save_path=fig_dir / "pca_variance.png")

    X_train_pca = transform_pca(pca, X_train)
    X_val_pca = transform_pca(pca, X_val)
    X_test_pca = transform_pca(pca, X_test)

    np.save(emb_dir / "train_X_pca.npy", X_train_pca)
    np.save(emb_dir / "val_X_pca.npy", X_val_pca)
    np.save(emb_dir / "test_X_pca.npy", X_test_pca)

    # UMAP 2D visualización
    logger.info("Ajustando UMAP 2D (visualización)...")
    umap_cfg = cfg.get("umap", {})
    reducer_2d = fit_umap(
        X_train,
        n_components=2,
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        metric=umap_cfg.get("metric", "cosine"),
        low_memory=umap_cfg.get("low_memory", True),
        seed=cfg["seed"],
    )
    X_train_2d = transform_umap(reducer_2d, X_train)
    X_test_2d = transform_umap(reducer_2d, X_test)

    plot_umap_2d(X_train_2d, y_train, title="UMAP 2D — Train (embeddings contrastivos)",
                 save_path=fig_dir / "umap_2d_train.png")
    plot_umap_2d(X_test_2d, y_test, title="UMAP 2D — Test (embeddings contrastivos)",
                 save_path=fig_dir / "umap_2d_test.png")

    # UMAP para clasificación (n_components > 2)
    umap_clf_dims = umap_cfg.get("n_components_clf", [50])
    umap_results = {}
    for n_dim in umap_clf_dims:
        logger.info(f"UMAP {n_dim}D para clasificación...")
        reducer = fit_umap(X_train, n_components=n_dim, seed=cfg["seed"],
                           n_neighbors=umap_cfg.get("n_neighbors", 15),
                           metric=umap_cfg.get("metric", "cosine"),
                           low_memory=True)
        Xtr = transform_umap(reducer, X_train)
        Xv = transform_umap(reducer, X_val)
        Xte = transform_umap(reducer, X_test)
        np.save(emb_dir / f"train_X_umap{n_dim}.npy", Xtr)
        np.save(emb_dir / f"val_X_umap{n_dim}.npy", Xv)
        np.save(emb_dir / f"test_X_umap{n_dim}.npy", Xte)
        umap_results[f"umap_{n_dim}d"] = {"shape": list(Xtr.shape)}

    save_json({"pca_n_components": n_comp, "umap": umap_results}, out_dir / "reduction_summary.json")

    # ── Reevaluación de top-2 modelos ─────────────────────────────────────────
    models_pkl = Path("artifacts/checkpoints/classical_models.pkl")
    if not models_pkl.exists():
        logger.warning("No se encontró classical_models.pkl — saltando reevaluación")
        return

    with open(models_pkl, "rb") as f:
        trained_models = pickle.load(f)

    top_models = cfg.get("top_models", [])
    reeval_results = {}
    bootstrap_cfg = {"n_resamples": 1000, "confidence_level": 0.95, "random_state": cfg["seed"]}

    for model_name in top_models:
        if model_name not in trained_models:
            logger.warning(f"Modelo '{model_name}' no encontrado")
            continue

        model = trained_models[model_name]
        reeval_results[model_name] = {}

        # Original (baseline)
        r = evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test,
                           f"{model_name}_original", bootstrap_cfg)
        reeval_results[model_name]["original_1024d"] = r["splits"]["test"]["accuracy"]

        # PCA
        r_pca = evaluate_model(model, X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test,
                               f"{model_name}_pca", bootstrap_cfg)
        reeval_results[model_name][f"pca_{n_comp}d"] = r_pca["splits"]["test"]["accuracy"]

        # UMAP (primer dim de clf)
        if umap_clf_dims:
            n_dim = umap_clf_dims[0]
            Xtr_u = np.load(emb_dir / f"train_X_umap{n_dim}.npy")
            Xv_u = np.load(emb_dir / f"val_X_umap{n_dim}.npy")
            Xte_u = np.load(emb_dir / f"test_X_umap{n_dim}.npy")
            r_umap = evaluate_model(model, Xtr_u, y_train, Xv_u, y_val, Xte_u, y_test,
                                    f"{model_name}_umap{n_dim}", bootstrap_cfg)
            reeval_results[model_name][f"umap_{n_dim}d"] = r_umap["splits"]["test"]["accuracy"]

        logger.info(f"{model_name}: {reeval_results[model_name]}")

    save_json(reeval_results, out_dir / "reevaluation_reduction.json")
    logger.info(f"Reevaluación guardada en {out_dir / 'reevaluation_reduction.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reduction.yaml")
    args = parser.parse_args()
    main(args.config)
