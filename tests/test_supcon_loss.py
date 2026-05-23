"""Tests: SupCon Loss — positivity, gradients, edge cases."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.contrastive_loss import SupConLoss


def _normalized(B=8, D=128, seed=0):
    torch.manual_seed(seed)
    f = torch.randn(B, D)
    return torch.nn.functional.normalize(f, dim=1)


def test_loss_positive():
    loss_fn = SupConLoss(temperature=0.07)
    features = _normalized(B=8)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss = loss_fn(features, labels)
    assert loss.item() > 0.0


def test_loss_gradient_flows():
    features = _normalized(B=8).requires_grad_(True)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    loss_fn = SupConLoss()
    loss = loss_fn(features, labels)
    loss.backward()
    assert features.grad is not None
    assert not torch.isnan(features.grad).any()


def test_loss_decreases_with_better_separation():
    """Embeddings perfectamente separados por clase deben tener menor pérdida."""
    loss_fn = SupConLoss(temperature=0.07)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    # Embeddings confusos (aleatorios)
    torch.manual_seed(0)
    f_random = torch.nn.functional.normalize(torch.randn(8, 128), dim=1)
    loss_random = loss_fn(f_random, labels).item()

    # Embeddings bien separados (clase 0 en mitad positiva, clase 1 en negativa)
    f_sep = torch.zeros(8, 128)
    f_sep[:4, 0] = 1.0   # clase 0 apunta en dirección +x
    f_sep[4:, 1] = 1.0   # clase 1 apunta en dirección +y
    f_sep = torch.nn.functional.normalize(f_sep, dim=1)
    loss_sep = loss_fn(f_sep, labels).item()

    assert loss_sep < loss_random, f"loss_sep={loss_sep:.4f} debe ser < loss_random={loss_random:.4f}"


def test_loss_raises_on_single_sample():
    loss_fn = SupConLoss()
    features = _normalized(B=1)
    labels = torch.tensor([0])
    with pytest.raises(ValueError):
        loss_fn(features, labels)


def test_loss_no_nan():
    loss_fn = SupConLoss()
    for _ in range(10):
        f = _normalized(B=16)
        labels = torch.randint(0, 2, (16,))
        loss = loss_fn(f, labels)
        assert not torch.isnan(loss), "Loss es NaN"
