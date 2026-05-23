"""
Entrena y evalúa los 5 modelos clásicos sobre los embeddings.

Uso:
    python -m scripts.train_models --config configs/classical.yaml

Prerequisito:
    - data/embeddings/{train,val,test}_X.npy deben existir
    - Ejecutar scripts/extract_embeddings.py primero
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.confusion import plot_confusion_matrix
from src.training.train_classical import train_single_model
from src.utils.io import load_config, load_embeddings, save_json
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("train_models", log_dir="artifacts/logs")


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    emb_dir = Path(cfg["embeddings_dir"])
    out_dir = Path(cfg["output_dir"])
    fig_dir = Path(cfg["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Cargar embeddings
    logger.info("Cargando embeddings...")
    X_train, y_train = load_embeddings("train", emb_dir)
    X_val, y_val = load_embeddings("val", emb_dir)
    X_test, y_test = load_embeddings("test", emb_dir)
    logger.info(f"  Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    models_cfg = cfg.get("models", {})
    bootstrap_cfg = cfg.get("bootstrap", {})
    cv_folds = cfg.get("cv_folds", 5)
    seed = cfg.get("seed", 42)

    all_results = {}
    trained_models = {}

    for model_name, model_cfg in models_cfg.items():
        if not model_cfg.get("enabled", True):
            logger.info(f"Saltando {model_name} (disabled)")
            continue

        try:
            results, model = train_single_model(
                model_name=model_name,
                model_cfg=model_cfg,
                X_train=X_train, y_train=y_train,
                X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test,
                cv_folds=cv_folds,
                seed=seed,
                bootstrap_cfg=bootstrap_cfg,
            )
        except Exception as e:
            logger.error(f"Error entrenando {model_name}: {e}")
            continue

        all_results[model_name] = results
        trained_models[model_name] = model

        # Guardar métricas individuales
        save_json(results, out_dir / f"{model_name}.json")

        # Matriz de confusión en test
        import numpy as np
        y_pred_test = model.predict(X_test)
        plot_confusion_matrix(
            y_test, y_pred_test,
            model_name=model_name,
            save_path=fig_dir / f"cm_{model_name}.png",
        )

        # Log resumen
        test_acc = results["splits"]["test"]["accuracy"]
        test_f1 = results["splits"]["test"]["f1_macro"]
        test_auc = results["splits"]["test"].get("roc_auc", float("nan"))
        logger.info(f"  {model_name:25s} | acc={test_acc:.4f} | f1={test_f1:.4f} | auc={test_auc:.4f}")

    # Guardar resumen comparativo
    summary = {
        name: {
            "test_accuracy": r["splits"]["test"]["accuracy"],
            "test_f1_macro": r["splits"]["test"]["f1_macro"],
            "test_roc_auc": r["splits"]["test"].get("roc_auc", None),
            "test_balanced_acc": r["splits"]["test"]["balanced_accuracy"],
            "train_time_s": r.get("train_time_s", None),
            "best_params": r.get("best_params", {}),
        }
        for name, r in all_results.items()
    }
    save_json(summary, out_dir / "summary.json")

    # Guardar modelos entrenados
    models_save_dir = Path("artifacts/checkpoints")
    models_save_dir.mkdir(parents=True, exist_ok=True)
    with open(models_save_dir / "classical_models.pkl", "wb") as f:
        pickle.dump(trained_models, f)

    logger.info(f"\nResumen guardado en {out_dir / 'summary.json'}")
    logger.info("Ranking por test F1-macro:")
    for name, s in sorted(summary.items(), key=lambda x: x[1]["test_f1_macro"], reverse=True):
        logger.info(f"  {name:25s}: {s['test_f1_macro']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classical.yaml")
    args = parser.parse_args()
    main(args.config)
