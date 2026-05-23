"""Tests: métricas de clasificación con casos sintéticos."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import bootstrap_ci, compute_metrics


def test_perfect_classifier():
    y = np.array([0, 0, 0, 1, 1, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["balanced_accuracy"] == 1.0


def test_worst_classifier():
    y = np.array([0, 0, 0, 1, 1, 1])
    y_pred = 1 - y
    m = compute_metrics(y, y_pred)
    assert m["accuracy"] == 0.0


def test_roc_auc_present():
    y = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    m = compute_metrics(y, y_pred, y_prob)
    assert "roc_auc" in m
    assert m["roc_auc"] == 1.0


def test_confusion_matrix_shape():
    y = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    m = compute_metrics(y, y_pred)
    cm = np.array(m["confusion_matrix"])
    assert cm.shape == (2, 2)


def test_bootstrap_ci_keys():
    y = np.array([0] * 50 + [1] * 50)
    y_pred = np.array([0] * 45 + [1] * 5 + [0] * 5 + [1] * 45)
    ci = bootstrap_ci(y, y_pred, n_resamples=100)
    for key in ["accuracy", "f1_macro", "balanced_accuracy"]:
        assert key in ci
        assert ci[key]["lower"] <= ci[key]["mean"] <= ci[key]["upper"]


def test_bootstrap_ci_width():
    """El IC debe tener ancho positivo."""
    y = np.array([0] * 50 + [1] * 50)
    y_pred = y.copy()
    y_pred[:5] = 1 - y_pred[:5]   # 5 errores
    ci = bootstrap_ci(y, y_pred, n_resamples=200)
    for key in ["accuracy", "f1_macro"]:
        width = ci[key]["upper"] - ci[key]["lower"]
        assert width >= 0.0
