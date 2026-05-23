"""Métricas de clasificación con intervalo de confianza bootstrap."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.stats import sem
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None = None) -> dict:
    """Calcula el conjunto completo de métricas para un split."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_per_class": f1_score(y_true, y_pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(y_true, y_pred, output_dict=True),
    }
    if y_prob is not None:
        prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        metrics["roc_auc"] = roc_auc_score(y_true, prob)
    return metrics


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, dict[str, float]]:
    """
    IC bootstrap para las métricas clave.
    Retorna {metric_name: {mean, lower, upper, std}}.
    """
    rng = np.random.default_rng(random_state)
    N = len(y_true)
    metric_samples: dict[str, list] = {k: [] for k in ["accuracy", "f1_macro", "balanced_accuracy", "roc_auc"]}

    for _ in range(n_resamples):
        idx = rng.integers(0, N, size=N)
        yt, yp = y_true[idx], y_pred[idx]
        metric_samples["accuracy"].append(accuracy_score(yt, yp))
        metric_samples["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        metric_samples["balanced_accuracy"].append(balanced_accuracy_score(yt, yp))
        if y_prob is not None:
            yprob = y_prob[idx]
            prob = yprob[:, 1] if yprob.ndim == 2 else yprob
            try:
                metric_samples["roc_auc"].append(roc_auc_score(yt, prob))
            except ValueError:
                metric_samples["roc_auc"].append(float("nan"))
        else:
            metric_samples["roc_auc"].append(float("nan"))

    alpha = 1 - confidence
    ci = {}
    for metric, samples in metric_samples.items():
        arr = np.array(samples)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            continue
        ci[metric] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "lower": float(np.percentile(arr, 100 * alpha / 2)),
            "upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
        }
    return ci


def evaluate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    bootstrap_cfg: dict | None = None,
) -> dict[str, Any]:
    """Evalúa un modelo sklearn en train/val/test y retorna dict completo de resultados."""
    bootstrap_cfg = bootstrap_cfg or {"n_resamples": 1000, "confidence_level": 0.95, "random_state": 42}

    results: dict[str, Any] = {"model": model_name, "splits": {}}
    t0 = time.time()

    for split_name, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y, y_pred, y_prob)

        if split_name == "test":
            ci = bootstrap_ci(
                y, y_pred, y_prob,
                n_resamples=bootstrap_cfg.get("n_resamples", 1000),
                confidence=bootstrap_cfg.get("confidence_level", 0.95),
                random_state=bootstrap_cfg.get("random_state", 42),
            )
            metrics["bootstrap_ci_95"] = ci

        results["splits"][split_name] = metrics

    results["inference_time_s"] = time.time() - t0
    return results
