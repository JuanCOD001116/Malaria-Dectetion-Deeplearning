# Malaria Detection — Contrastive Learning + Classical Models

Proyecto académico de clasificación binaria de malaria usando:
1. **Encoder CNN** entrenado con **Supervised Contrastive Learning (SupCon)** → embeddings de 1024 dims
2. **5 modelos clásicos** comparados sobre los mismos embeddings
3. **PCA + UMAP** para análisis de reducción de dimensión

Dataset: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) — 27,558 imágenes balanceadas (50/50).

## Metodología

```
Imágenes → ResNet18+SupCon → Embeddings 1024D → {LR, KNN, RF, MLP, SVM}
```

- La CNN **no** clasifica directamente. Se entrena para **similitud** (SupCon loss).
- Todos los modelos usan **exactamente el mismo vector de 1024 features**.
- Split estratificado fijo (`seed=42`). Sin oversampling.

## Setup rápido

```powershell
# 1. Clonar y entrar al repo
git clone https://github.com/TU_USUARIO/Malaria-Dectetion-Deeplearning.git
cd Malaria-Dectetion-Deeplearning

# 2. Crear entorno virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Preparar datos (genera splits en data/processed/)
python -m scripts.prepare_data --config configs/data.yaml
```

> El dataset `cell_images/` debe estar en la raíz del repo (ya incluido).

## Ejecución del pipeline completo

### Paso 1: EDA
```powershell
jupyter lab notebooks/01_eda.ipynb
```

### Paso 2: Entrenamiento contrastivo
**Opción A — Colab (recomendado, GPU):**
- Abre `notebooks/02_contrastive_training.ipynb` en Google Colab
- Runtime → Change runtime type → T4 GPU
- Descomenta las celdas `# COLAB` y ejecuta todo
- Descarga `encoder_best.pt` a `artifacts/checkpoints/`

**Opción B — Local (CPU, lento ~2h/epoch):**
```powershell
python -m scripts.train_contrastive --config configs/contrastive.yaml
```

### Paso 3: Extracción de embeddings
```powershell
python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt
# Genera: data/embeddings/{train,val,test}_{X,y}.npy
```

### Paso 4: Modelos clásicos
```powershell
python -m scripts.train_models --config configs/classical.yaml
# Genera: artifacts/metrics/{model_name}.json + artifacts/figures/cm_*.png
```

### Paso 5: Reducción de dimensión + Similitud
```powershell
python -m scripts.run_reduction --config configs/reduction.yaml
python -m scripts.evaluate_similarity
```

### Paso 6: Evaluación final
```powershell
jupyter lab notebooks/06_final_evaluation.ipynb
```

## Tests

```powershell
# Tests unitarios
pytest tests/ -v

# Smoke test E2E (con datos sintéticos, <2 min)
pytest tests/test_pipeline_smoke.py -v
```

## Estructura del proyecto

```
├── configs/          → Hiperparámetros centralizados (YAML)
├── data/
│   ├── processed/    → Splits CSV (train/val/test)
│   └── embeddings/   → .npy de embeddings 1024D
├── src/
│   ├── data/         → Dataset, split, augmentations
│   ├── models/       → ContrastiveEncoder (ResNet18+SupCon)
│   ├── training/     → SupConLoss, loops de entrenamiento
│   ├── evaluation/   → Métricas, confusión, similitud
│   ├── reduction/    → PCA, UMAP, análisis de features
│   ├── visualization/→ Plots EDA, entrenamiento, embeddings
│   └── utils/        → Seed, I/O, logging
├── scripts/          → CLIs ejecutables por línea de comandos
├── notebooks/        → 6 notebooks (01-06), sin ejecutar
├── tests/            → pytest — unitarios + smoke E2E
└── artifacts/        → checkpoints, logs, figuras, métricas
```

## Reproducibilidad

- `seed=42` en todos los configs y scripts (`src/utils/seed.py`)
- Splits CSV fijos en `data/processed/` (commiteados en git)
- `torch.backends.cudnn.deterministic = True`
- Resultados reportados: train/val/test + IC 95% bootstrap

## Requisitos de compute

| Tarea | CPU local | Colab T4 (gratis) |
|---|---|---|
| EDA | ~2 min | ~2 min |
| Entrenamiento encoder (30 epochs) | ~6h | ~15-30 min |
| Extracción embeddings | ~5 min | ~3 min |
| Modelos clásicos | ~30 min | ~20 min |
| PCA + UMAP | ~10 min | ~5 min |

## Ver también

- [`docs/EJECUCION.md`](docs/EJECUCION.md) — Guía detallada y sincronización Colab-VSCode
- [`AGENTS.md`](AGENTS.md) — Plan metodológico completo
