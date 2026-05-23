"""
Entrena los 5 modelos clásicos sobre embeddings con GridSearchCV.
Todos usan exactamente los mismos embeddings — comparación justa.
"""
from __future__ import annotations

import importlib
import time
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import evaluate_model
from src.utils.logging import get_logger

logger = get_logger("train_classical")


def _build_estimator(model_cfg: dict, seed: int) -> Any:
    """Instancia un estimador sklearn desde su class string."""
    module_path, class_name = model_cfg["class"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    fixed_params = model_cfg.get("fixed_params", {})
    # Inyectar random_state si el estimador lo acepta
    try:
        return cls(random_state=seed, **fixed_params)
    except TypeError:
        return cls(**fixed_params)


def train_single_model(
    model_name: str,
    model_cfg: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int,
    seed: int,
    bootstrap_cfg: dict,
) -> dict[str, Any]:
    """Entrena y evalúa un modelo con GridSearchCV sobre train+val."""
    logger.info(f"Entrenando: {model_name}")

    estimator = _build_estimator(model_cfg, seed)

    # Pipeline: normalización + modelo
    # StandardScaler es importante para SVM, LR, KNN
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", estimator),
    ])

    # Construir param_grid con prefijo "clf__"
    raw_grid = model_cfg.get("grid", {})
    param_grid = {f"clf__{k}": v for k, v in raw_grid.items()}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # GridSearchCV sobre train+val combinados
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])

    t0 = time.time()
    if param_grid:
        search = GridSearchCV(
            pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0, refit=True
        )
        search.fit(X_trainval, y_trainval)
        best_model = search.best_estimator_
        best_params = search.best_params_
        cv_score = search.best_score_
    else:
        pipe.fit(X_trainval, y_trainval)
        best_model = pipe
        best_params = {}
        cv_score = 0.0

    train_time = time.time() - t0
    logger.info(f"  Mejor params: {best_params}")
    logger.info(f"  CV F1-macro: {cv_score:.4f} | Tiempo: {train_time:.1f}s")

    results = evaluate_model(
        model=best_model,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        model_name=model_name,
        bootstrap_cfg=bootstrap_cfg,
    )
    results["best_params"] = best_params
    results["cv_f1_macro"] = cv_score
    results["train_time_s"] = train_time

    return results, best_model
