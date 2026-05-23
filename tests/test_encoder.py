"""Tests: forward pass, shapes y gradientes del encoder."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.encoder import ContrastiveEncoder


def _make_batch(B=4, C=3, H=96, W=96):
    return torch.randn(B, C, H, W)


def test_encoder_embedding_shape():
    model = ContrastiveEncoder(embedding_dim=1024, proj_dim=128, pretrained=False)
    model.eval()
    x = _make_batch(B=4)
    with torch.no_grad():
        z = model(x, return_embedding=True)
    assert z.shape == (4, 1024), f"Expected (4, 1024), got {z.shape}"


def test_encoder_projection_shape():
    model = ContrastiveEncoder(embedding_dim=1024, proj_dim=128, pretrained=False)
    model.eval()
    x = _make_batch(B=4)
    with torch.no_grad():
        p = model(x, return_embedding=False)
    assert p.shape == (4, 128), f"Expected (4, 128), got {p.shape}"


def test_encoder_projection_l2_normalized():
    """La proyección contrastiva debe estar l2-normalizada."""
    model = ContrastiveEncoder(pretrained=False)
    model.eval()
    x = _make_batch(B=8)
    with torch.no_grad():
        p = model(x)
    norms = torch.norm(p, dim=1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), f"Normas: {norms}"


def test_encoder_gradient_flows():
    """La pérdida de SupCon debe retropropagar sin errores."""
    model = ContrastiveEncoder(pretrained=False)
    model.train()
    x = _make_batch(B=4)
    p = model(x)
    loss = (1 - p @ p.T).mean()
    loss.backward()
    # Verificar que al menos un parámetro tiene gradiente
    has_grad = any(param.grad is not None for param in model.parameters())
    assert has_grad


def test_get_embedding_alias():
    model = ContrastiveEncoder(pretrained=False)
    model.eval()
    x = _make_batch(B=2)
    z = model.get_embedding(x)
    assert z.shape == (2, 1024)
