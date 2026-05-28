# Malaria Detection — Contrastive Learning + Modelos Clásicos

**URL DEL VIDEO DE EXPLICACION:** [LINK DEL VIDEO VIDEO](https://youtu.be/tsSfXmz2XoQ)

Clasificación binaria de imágenes de células de malaria (*Parasitized* vs *Uninfected*) en dos fases:

1. **Encoder ResNet18** entrenado con **Supervised Contrastive Learning (SupCon)** → embeddings de 1024 dimensiones.
2. **5 modelos clásicos** (LR, KNN, RF, MLP, SVM) entrenados sobre los mismos embeddings, comparados con bootstrap CI 95 %.
3. **PCA + UMAP** para reducción de dimensión y análisis de similitud coseno intra/inter-clase.

Dataset: [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) — 27 558 imágenes balanceadas 50/50.

```
Imágenes (3×96×96) ──► ResNet18 + SupCon ──► Embeddings (1024-d) ──► {LR, KNN, RF, MLP, SVM}
                                                  │
                                                  └──► PCA / UMAP / similitud coseno
```

## Resultados resumen (test, $N = 4\,135$)

| Modelo | Test Acc | Test $F_1$ macro | Test AUC | CI 95 % (Acc) |
|--------|---------:|-----------------:|---------:|---------------|
| **MLP**            | **0.9703** | **0.9703** | **0.9932** | [0.9649, 0.9758] |
| **SVM** (lineal)   | **0.9703** | **0.9703** | n/a        | [0.9644, 0.9756] |
| KNN ($k=11$, coseno) | 0.9700 | 0.9700 | 0.9852 | [0.9640, 0.9753] |
| Regresión logística  | 0.9698 | 0.9698 | 0.9931 | [0.9642, 0.9753] |
| Random Forest        | 0.9688 | 0.9688 | 0.9928 | [0.9635, 0.9741] |

Reporte académico completo (Metodología + Resultados, con todas las figuras y la matemática del SupCon loss) en [docs/REPORTE_ACADEMICO.md](docs/REPORTE_ACADEMICO.md).

---

## Tabla de contenidos

- [Requisitos previos](#requisitos-previos)
- [Ruta A — Reproducción 100 % en Google Colab](#ruta-a--reproducción-100--en-google-colab-recomendada)
- [Ruta B — Reproducción local (CPU/GPU)](#ruta-b--reproducción-local-cpugpu)
- [Ruta C — Híbrida (encoder en Colab, resto local)](#ruta-c--híbrida-encoder-en-colab-resto-local)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Archivos de configuración](#archivos-de-configuración)
- [Reproducibilidad](#reproducibilidad)
- [Tests](#tests)
- [Solución de problemas](#solución-de-problemas)

---

## Requisitos previos

| Recurso | Versión / detalle |
|---------|-------------------|
| Python | 3.10 – 3.12 (probado en 3.11) |
| Sistema | Windows / Linux / macOS |
| GPU (opcional) | NVIDIA con CUDA ≥ 11.8 — sólo necesaria para entrenar el encoder |
| Dataset | Cuenta de Kaggle + `kaggle.json` (API token) |
| Si se reproduce en Colab | Cuenta de Google con Drive (≥ 1 GB libre) |

> El dataset (≈ 340 MB descomprimido) **no** se versiona en git. Se descarga automáticamente mediante la API de Kaggle.

### Tiempos esperados

| Fase | CPU local | Colab T4 (gratis) |
|---|---:|---:|
| EDA (notebook 01) | ~2 min | ~2 min |
| Entrenamiento encoder (50 epochs, ES en ~30) | ~6 h | **~20–30 min** |
| Extracción de embeddings | ~5 min | ~3 min |
| Modelos clásicos (5 modelos + GridSearch) | ~10 min | ~8 min |
| PCA + UMAP + similitud | ~10 min | ~5 min |
| **Total** | ~7 h | **~1 h** |

---

## Ruta A — Reproducción 100 % en Google Colab (recomendada)

Los 6 notebooks autodetectan Colab y configuran automáticamente: clonan el repo, montan Drive para artefactos pesados, descargan el dataset desde Kaggle y persisten checkpoints/embeddings entre notebooks vía *symlinks*.

### A.1. Configuración única (una sola vez, ~5 minutos)

#### Paso 1 — Generar tokens

- **Kaggle API:** [`kaggle.com/settings`](https://www.kaggle.com/settings) → *Create New API Token* → descarga `kaggle.json`.
- **GitHub (opcional):** sólo si quieres hacer push automático de los notebooks ejecutados. No es necesario para reproducir los resultados — el notebook actual ofrece descarga manual al final.

#### Paso 2 — Abrir el primer notebook

1. Abre Colab → **File → Open notebook → GitHub** → pega la URL del repo:
   `https://github.com/JuanCOD001116/Malaria-Dectetion-Deeplearning`
2. Selecciona `notebooks/01_eda.ipynb`.
3. **Runtime → Change runtime type → CPU** (el EDA no necesita GPU).
4. **Runtime → Run all**.
5. La primera celda pedirá:
   - Autorizar acceso a Google Drive (se montará en `/content/drive`).
   - Subir `kaggle.json` (se cachea en Drive → no hay que volver a subirlo en los siguientes notebooks).

### A.2. Ejecutar los 6 notebooks en orden

| # | Notebook | Runtime | Tiempo | Artefactos generados |
|---|----------|---------|-------:|----------------------|
| 1 | `01_eda.ipynb` | CPU | ~2 min | `data/processed/{train,val,test}.csv`, 5 figuras EDA |
| 2 | `02_contrastive_training.ipynb` | **T4 GPU** | ~25 min | `encoder_best.pt` (Drive), `training_curves.png` |
| 3 | `03_extract_embeddings.ipynb` | T4 o CPU | ~4 min | `{train,val,test}_{X,y}.npy` en Drive |
| 4 | `04_classical_models.ipynb` | **T4 GPU** | ~8 min | 5 × `*.json`, 5 × `cm_*.png`, `models_comparison.png`, `roc_curves.png` |
| 5 | `05_reduction_and_similarity.ipynb` | CPU | ~6 min | PCA/UMAP figuras, `nearest_neighbors.json`, `reevaluation_reduction.json` |
| 6 | `06_final_evaluation.ipynb` | CPU | ~1 min | `final_comparison_table.csv`, `final_metrics_comparison.png` |

Para cada notebook: **abrir desde GitHub → seleccionar runtime → Run all**.

> **Sólo los notebooks 02 y 04 requieren GPU.** Cambia el runtime a **T4 GPU** antes de ejecutarlo. El resto puede correr en CPU.

### A.3. Verificación tras cada notebook

- **NB01:** debe imprimir `Train: 19290 | Val: 4133 | Test: 4135` y `0 imágenes corruptas`.
- **NB02:** debe imprimir `✓ Checkpoint en Drive: ... (147.7 MB)` al final.
- **NB03:** debe imprimir shapes `(19290, 1024)`, `(4133, 1024)`, `(4135, 1024)`.
- **NB04:** el ranking final debe mostrar accuracy de test en torno a 0.97 para los 5 modelos.
- **NB05:** debe imprimir `Separability gap ≈ 0.8170`.
- **NB06:** debe imprimir la tabla comparativa final con los 5 modelos.

### A.4. Persistencia entre sesiones

Si Colab se desconecta a mitad de un notebook:

1. Reabre el mismo notebook en Colab.
2. **Runtime → Run all**.
3. La celda de setup detecta que el repo y Drive ya existen y sólo hace `git pull` + remonta Drive.
4. Los artefactos ya generados (checkpoint, embeddings) se reutilizan sin reentrenar — viven en Drive vía *symlink*.

Estructura en Drive tras NB02 y NB03:

```
/content/drive/MyDrive/malaria_project/
├── kaggle.json                 # cacheado tras NB01
├── checkpoints/
│   ├── encoder_best.pt         # NB02  (~147 MB)
│   ├── encoder_last.pt         # NB02
│   └── classical_models.pkl    # NB04
└── embeddings/
    ├── train_X.npy, train_y.npy    # NB03  (~80 MB total)
    ├── val_X.npy,   val_y.npy
    └── test_X.npy,  test_y.npy
```

### A.5. Descargar los notebooks ejecutados

Al final de cada notebook hay una celda que descarga el `.ipynb` ejecutado a tu PC. Súbelos manualmente a tu repositorio (no hay auto-push activado por seguridad).

---

## Ruta B — Reproducción local (CPU/GPU)

### B.1. Clonar e instalar

```powershell
# Windows PowerShell — para Linux/macOS usa los equivalentes (source .venv/bin/activate, etc.)
git clone https://github.com/JuanCOD001116/Malaria-Dectetion-Deeplearning.git
Set-Location Malaria-Dectetion-Deeplearning

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

Alternativa con conda:

```powershell
conda env create -f environment.yml
conda activate malaria-detection
```

**Verificar instalación:**

```powershell
python -c "import torch, torchvision, sklearn, umap; print('OK')"
pytest tests/test_pipeline_smoke.py -v       # < 2 min, valida E2E con datos sintéticos
```

### B.2. Obtener el dataset

```powershell
# Opción 1 — Kaggle API (recomendada)
pip install kaggle
# Coloca tu kaggle.json en %USERPROFILE%\.kaggle\
kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
Expand-Archive cell-images-for-detecting-malaria.zip -DestinationPath .
# Si la descarga crea cell_images/cell_images/, mueve el contenido un nivel arriba:
# Move-Item cell_images\cell_images\* cell_images\; Remove-Item cell_images\cell_images
```

Tras esto debes tener:

```
cell_images/
├── Parasitized/   # 13 779 PNG
└── Uninfected/    # 13 779 PNG
```

### B.3. Pipeline paso a paso

Cada paso es un script CLI ejecutable con `python -m`. Los configs por defecto están en `configs/` — no es necesario pasarlos explícitamente, los muestro aquí por claridad.

```powershell
# Paso 1 — Generar splits estratificados (~30 s)
python -m scripts.prepare_data --config configs/data.yaml
# Salida: data/processed/{train,val,test}.csv

# Paso 2 — Entrenar encoder contrastivo
#   ATENCIÓN: en CPU tarda ~6 h. Si tienes GPU local con CUDA, autodetecta.
#   Para iteración rápida edita configs/contrastive.yaml y reduce training.epochs.
python -m scripts.train_contrastive --config configs/contrastive.yaml
# Salida: artifacts/checkpoints/encoder_best.pt, artifacts/logs/contrastive_history.json

# Paso 3 — Extraer embeddings 1024-d (CPU ~5 min, GPU ~2 min)
python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt
# Salida: data/embeddings/{train,val,test}_{X,y}.npy

# Paso 4 — Entrenar y evaluar los 5 modelos clásicos (~10 min)
python -m scripts.train_models --config configs/classical.yaml
# Salida: artifacts/metrics/{lr,knn,rf,mlp,svm}.json + artifacts/figures/cm_*.png
#         + artifacts/checkpoints/classical_models.pkl

# Paso 5 — PCA + UMAP + reevaluación de top-2 modelos (~10 min)
python -m scripts.run_reduction --config configs/reduction.yaml
# Salida: artifacts/figures/{pca_variance, umap_2d_*, feature_analysis}.png
#         + artifacts/metrics/{reduction_summary, reevaluation_reduction}.json

# Paso 6 — Análisis de similitud coseno (~2 min)
python -m scripts.evaluate_similarity
# Salida: artifacts/figures/{cosine_sim_distributions, cosine_sim_heatmap}.png
#         + artifacts/metrics/{nearest_neighbors, similarity_summary}.json
```

### B.4. Verificación final

```powershell
# Debe imprimir 5 JSONs con métricas de test
Get-ChildItem artifacts\metrics\*.json | ForEach-Object { $_.Name }

# Debe imprimir 22 PNGs
(Get-ChildItem artifacts\figures\*.png).Count

# Tabla final consolidada (opcional)
jupyter lab notebooks\06_final_evaluation.ipynb
```

---

## Ruta C — Híbrida (encoder en Colab, resto local)

La fase 2 (entrenar el encoder) es la única que se beneficia mucho de GPU. Esta ruta entrena en Colab y termina localmente.

```powershell
# (Local) Paso 1 — Setup y splits
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m scripts.prepare_data --config configs/data.yaml
```

Luego en Colab:

1. Abre `notebooks/02_contrastive_training.ipynb` desde GitHub.
2. Runtime → **T4 GPU** → Run all (≈25 min).
3. La celda final ofrece `files.download('artifacts/checkpoints/encoder_best.pt')` o copia a tu Drive.

De vuelta en local, coloca `encoder_best.pt` en `artifacts/checkpoints/` y continúa:

```powershell
python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt
python -m scripts.train_models --config configs/classical.yaml
python -m scripts.run_reduction --config configs/reduction.yaml
python -m scripts.evaluate_similarity
```

---

## Estructura del repositorio

```
.
├── README.md                       # este archivo
├── requirements.txt                # dependencias pip
├── environment.yml                 # dependencias conda
├── pytest.ini                      # configuración de tests
│
├── configs/                        # hiperparámetros (YAML, versionados)
│   ├── data.yaml                   # splits, normalización, paths
│   ├── contrastive.yaml            # encoder + SupCon + entrenamiento
│   ├── classical.yaml              # 5 modelos + grids + bootstrap
│   └── reduction.yaml              # PCA + UMAP + análisis features
│
├── data/
│   ├── processed/                  # splits CSV (versionados — reproducibilidad)
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── embeddings/                 # .npy (gitignored, ~80 MB)
│
├── src/                            # código modular
│   ├── data/                       # Dataset, split, augmentations
│   ├── models/encoder.py           # ContrastiveEncoder (ResNet18 + heads)
│   ├── training/
│   │   ├── contrastive_loss.py     # SupConLoss (Khosla et al. 2020)
│   │   ├── train_contrastive.py    # loop de entrenamiento SupCon
│   │   └── train_classical.py      # GridSearchCV + evaluación
│   ├── evaluation/                 # métricas, bootstrap, confusion, similitud
│   ├── reduction/                  # PCA, UMAP, feature analysis
│   ├── visualization/              # plots EDA / training / embeddings
│   └── utils/                      # seed, I/O, logging
│
├── scripts/                        # CLIs ejecutables (python -m scripts.X)
│   ├── prepare_data.py
│   ├── train_contrastive.py
│   ├── extract_embeddings.py
│   ├── train_models.py
│   ├── run_reduction.py
│   └── evaluate_similarity.py
│
├── notebooks/                      # 6 notebooks (autodetectan Colab vs local)
│   ├── 01_eda.ipynb
│   ├── 02_contrastive_training.ipynb
│   ├── 03_extract_embeddings.ipynb
│   ├── 04_classical_models.ipynb
│   ├── 05_reduction_and_similarity.ipynb
│   └── 06_final_evaluation.ipynb
│
├── tests/                          # pytest (36 tests, smoke E2E < 9 s)
│
├── artifacts/                      # outputs (parcialmente gitignored)
│   ├── checkpoints/                # .pt y .pkl  (gitignored)
│   ├── logs/                       # .json y .log (gitignored)
│   ├── figures/                    # 22 PNG (versionados)
│   └── metrics/                    # JSON y CSV con métricas (versionados)
│
└── docs/

```

---

## Archivos de configuración

Todos los hiperparámetros del proyecto viven en YAMLs versionados en `configs/`. Si quieres reproducir variantes (otro tamaño de imagen, otro batch, otra temperatura SupCon), edita estos archivos — no toques el código.

| Archivo | Controla |
|---------|----------|
| `configs/data.yaml` | `seed`, `img_size`, proporciones del split, normalización |
| `configs/contrastive.yaml` | Arquitectura del encoder (1024 / 128), $\tau$ SupCon, optimizer, scheduler, augmentations, batch sizes GPU/CPU |
| `configs/classical.yaml` | 5 modelos + sus grids `GridSearchCV`, `cv_folds`, bootstrap `n_resamples` y `confidence_level` |
| `configs/reduction.yaml` | Umbral de varianza PCA, hiperparámetros UMAP, top-K features, modelos a reevaluar |

Valores por defecto verificados:
`seed = 42` · `img_size = 96` · `embedding_dim = 1024` · `proj_dim = 128` · `temperature = 0.07` · `epochs = 50` · `batch_size_gpu = 256` · `early_stopping_patience = 7` · `cv_folds = 3` · `bootstrap.n_resamples = 200`.

---

## Reproducibilidad

El proyecto está diseñado para producir resultados bit-a-bit reproducibles cuando se respeta el mismo hardware/seed. Mecanismos en juego:

- `src/utils/seed.py::set_global_seed(42)` se invoca al inicio de cada script y notebook. Fija `random`, `numpy`, `torch`, `PYTHONHASHSEED`, y activa `cudnn.deterministic = True`.
- Los splits CSV están **versionados en git** ([data/processed/train.csv](data/processed/train.csv), `val.csv`, `test.csv`). Quien clone el repo obtiene exactamente los mismos índices.
- Las búsquedas `GridSearchCV` usan `StratifiedKFold(shuffle=True, random_state=42)`.
- El bootstrap usa `np.random.default_rng(42)`.

> **Determinismo entre GPU y CPU.** Algunos operadores de PyTorch en GPU son no determinísticos por diseño (e.g. ciertos backwards en convoluciones). Esperar diferencias del orden de $10^{-4}$ en pérdidas y $\pm 0.1\,\text{pp}$ en accuracy entre runs en hardware distinto. Los resultados reportados en [docs/REPORTE_ACADEMICO.md](docs/REPORTE_ACADEMICO.md) se obtuvieron en Colab Tesla T4.

---

## Tests

```powershell
# Tests unitarios completos (36 tests)
pytest tests/ -v

# Smoke E2E con datos sintéticos (< 9 s) — ideal pre-PR
pytest tests/test_pipeline_smoke.py -v

# Con cobertura
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Solución de problemas

### `ModuleNotFoundError: No module named 'src'`
Ejecuta los scripts desde la raíz del repo con la sintaxis `python -m scripts.X`, no `python scripts/X.py`. El `python -m` añade el cwd al `PYTHONPATH`.

### `num_workers > 0` cuelga en Windows
Ya está manejado: `scripts/train_contrastive.py` fuerza `num_workers = 0` en CPU. Si quieres usarlo en GPU local Windows y falla, edita `configs/contrastive.yaml` y pon `num_workers: 0`.

### `CUDA out of memory` en Colab
Reduce `batch_size_gpu` en `configs/contrastive.yaml` (256 → 128 → 64) o `img_size` (96 → 64). El SupCon escala bien con batches más pequeños.

### Kaggle: `403 Forbidden`
Vincula tu cuenta y acepta las reglas del dataset una vez en `kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria` desde el navegador, luego reintenta.

### `umap-learn` no instalable
```powershell
pip install umap-learn   # NO "umap" — ese paquete es otro
```

### Resultados ligeramente distintos a los reportados
Esperable si entrenaste en hardware distinto (CPU vs GPU, T4 vs A100). Las diferencias deben mantenerse < 0.2 pp en accuracy. Si superan eso, revisa que `seed = 42` esté propagado y que la versión de `torch` esté en `>= 2.1`.

### Colab desconecta a mitad del entrenamiento (NB02)
El loop guarda `encoder_last.pt` al final de cada epoch. Al reabrir el notebook, los symlinks a Drive ya tienen el último checkpoint — re-ejecutar la celda de entrenamiento continúa desde donde quedó.

---

## Licencia y citación

Proyecto académico — Materia *Modelos II*.

