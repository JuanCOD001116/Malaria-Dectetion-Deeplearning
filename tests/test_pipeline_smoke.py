"""
Smoke test E2E: corre el pipeline completo con un mini-dataset sintético.
Valida shapes y tipos en cada etapa sin entrenamiento real.
Debe completarse en < 2 min en CPU.

Uso:
    pytest tests/test_pipeline_smoke.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentations import get_contrastive_transform, get_eval_transform
from src.data.dataset import MalariaDataset, SupConPairDataset
from src.data.split import make_stratified_split
from src.evaluation.metrics import bootstrap_ci, compute_metrics
from src.models.encoder import ContrastiveEncoder
from src.training.contrastive_loss import SupConLoss
from src.utils.io import save_embeddings, load_embeddings
from src.utils.seed import set_global_seed


@pytest.fixture(scope="module")
def smoke_dataset(tmp_path_factory):
    """Crea 100 imágenes sintéticas (50/50) para el smoke test."""
    tmp = tmp_path_factory.mktemp("smoke")
    for cls, label in [("Parasitized", 1), ("Uninfected", 0)]:
        cls_dir = tmp / cls
        cls_dir.mkdir()
        for i in range(50):
            arr = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
            Image.fromarray(arr).save(cls_dir / f"img_{i:03d}.png")
    return tmp


@pytest.fixture(scope="module")
def smoke_splits(smoke_dataset, tmp_path_factory):
    proc_dir = tmp_path_factory.mktemp("processed")
    train_df, val_df, test_df = make_stratified_split(
        dataset_root=smoke_dataset,
        processed_dir=proc_dir,
        train_frac=0.70, val_frac=0.15, seed=42,
    )
    return proc_dir, train_df, val_df, test_df


def test_smoke_split(smoke_splits):
    proc_dir, train_df, val_df, test_df = smoke_splits
    total = len(train_df) + len(val_df) + len(test_df)
    assert total == 100


def test_smoke_dataset_load(smoke_splits):
    proc_dir, train_df, val_df, test_df = smoke_splits
    ds = MalariaDataset(proc_dir / "train.csv", transform=get_eval_transform(96))
    img, label = ds[0]
    assert img.shape == (3, 96, 96)
    assert label in (0, 1)


def test_smoke_supcon_dataset(smoke_splits):
    proc_dir, *_ = smoke_splits
    ds = SupConPairDataset(proc_dir / "train.csv", transform=get_contrastive_transform())
    v1, v2, label = ds[0]
    assert v1.shape == v2.shape == (3, 96, 96)


def test_smoke_encoder_forward():
    set_global_seed(42)
    model = ContrastiveEncoder(embedding_dim=1024, proj_dim=128, pretrained=False)
    model.eval()
    x = torch.randn(4, 3, 96, 96)
    with torch.no_grad():
        emb = model(x, return_embedding=True)
        proj = model(x, return_embedding=False)
    assert emb.shape == (4, 1024)
    assert proj.shape == (4, 128)
    norms = torch.norm(proj, dim=1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_smoke_supcon_loss():
    loss_fn = SupConLoss(temperature=0.07)
    features = torch.nn.functional.normalize(torch.randn(16, 128), dim=1)
    labels = torch.randint(0, 2, (16,))
    loss = loss_fn(features, labels)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_smoke_one_train_step():
    set_global_seed(42)
    model = ContrastiveEncoder(pretrained=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = SupConLoss()

    x = torch.randn(8, 3, 96, 96)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    views = torch.cat([x, x], dim=0)
    labels_rep = torch.cat([labels, labels])

    optimizer.zero_grad()
    proj = model(views)
    loss = loss_fn(proj, labels_rep)
    loss.backward()
    optimizer.step()
    assert loss.item() > 0


def test_smoke_embedding_save_load(tmp_path):
    X = np.random.randn(20, 1024).astype(np.float32)
    y = np.array([0] * 10 + [1] * 10)
    save_embeddings(X, y, "test", tmp_path)
    X2, y2 = load_embeddings("test", tmp_path)
    assert np.allclose(X, X2)
    assert np.array_equal(y, y2)


def test_smoke_metrics():
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1, 0, 1])
    m = compute_metrics(y_true, y_pred)
    assert 0 <= m["accuracy"] <= 1
    assert 0 <= m["f1_macro"] <= 1
    ci = bootstrap_ci(y_true, y_pred, n_resamples=50)
    assert "accuracy" in ci


def test_smoke_sklearn_model(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    X_tr = np.random.randn(60, 1024).astype(np.float32)
    y_tr = np.array([0] * 30 + [1] * 30)
    X_te = np.random.randn(20, 1024).astype(np.float32)
    y_te = np.array([0] * 10 + [1] * 10)

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=100))])
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    assert preds.shape == (20,)
    assert set(preds).issubset({0, 1})


def test_smoke_pca():
    from src.reduction.pca import fit_pca, transform_pca
    X = np.random.randn(50, 1024).astype(np.float32)
    pca, n_comp = fit_pca(X, variance_threshold=0.90, seed=42)
    X_red = transform_pca(pca, X)
    assert X_red.shape[1] == n_comp
    assert X_red.shape[1] < 1024
