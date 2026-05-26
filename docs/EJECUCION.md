# Guía de Ejecución Detallada + Sincronización Colab-VSCode

## Índice
0. [Ejecución 100% en Colab (recomendado para reporte IEEE)](#0-ejecución-100-en-colab-recomendado-para-reporte-ieee)
1. [Setup inicial](#1-setup-inicial)
2. [Flujo completo local (CPU)](#2-flujo-completo-local-cpu)
3. [Flujo con Colab para entrenamiento GPU](#3-flujo-con-colab-para-entrenamiento-gpu)
4. [Sincronización Colab ↔ VSCode / GitHub](#4-sincronización-colab--vscode--github)
5. [Ejecutar notebooks en GitHub con outputs](#5-ejecutar-notebooks-en-github-con-outputs)
6. [Solución de problemas comunes](#6-solución-de-problemas-comunes)

---

## 0. Ejecución 100% en Colab (recomendado para reporte IEEE)

Los 6 notebooks ya están adaptados para ejecutarse end-to-end en Colab. Cada uno:
- Auto-detecta el entorno (local vs Colab) y ejecuta el setup que corresponda
- Persiste artefactos pesados (`encoder_best.pt`, embeddings `.npy`, `classical_models.pkl`) en Google Drive
- Hace `git push` automático del notebook ejecutado + figuras/métricas/logs a GitHub

### Pre-requisitos (una sola vez, ~10 min)

#### A. Personal Access Token de GitHub
1. Ir a https://github.com/settings/tokens → **Generate new token (classic)**
2. Marcar scope `repo`
3. Copiar el token (`ghp_...`)

#### B. API token de Kaggle
1. https://www.kaggle.com/settings → **Create New API Token**
2. Guardar el `kaggle.json` descargado

#### C. Configurar Colab Secrets
En cualquier notebook abierto en Colab: **🔑 Secrets** (panel izquierdo) → agregar:

| Nombre | Valor |
|---|---|
| `GITHUB_TOKEN` | el token del paso A (`ghp_...`) |
| `GITHUB_USER` | tu usuario de GitHub (ej. `JuanCOD001116`) |
| `GITHUB_EMAIL` | tu email del git config |

> Marca **Notebook Access: ON** para los tres. Los secrets se guardan a nivel de cuenta Google y persisten entre sesiones.

#### D. Estructura en Drive (se crea automáticamente)

```
/content/drive/MyDrive/malaria_project/
├── kaggle.json                          # cacheado tras el primer NB01
├── checkpoints/
│   ├── encoder_best.pt                  # NB02
│   ├── encoder_last.pt                  # NB02
│   └── classical_models.pkl             # NB04
└── embeddings/
    ├── train_X.npy, train_y.npy         # NB03
    ├── val_X.npy, val_y.npy
    └── test_X.npy, test_y.npy
```

### Orden de ejecución

Cada notebook se abre desde **File → Open notebook → GitHub** → URL del repo → seleccionar el `.ipynb`.

| Paso | Notebook | Runtime | Duración | Output principal |
|---|---|---|---|---|
| 1 | `01_eda.ipynb` | CPU | ~5 min | splits CSV + 6 figuras EDA |
| 2 | `02_contrastive_training.ipynb` | **T4 GPU** | ~20-30 min | `encoder_best.pt` → Drive |
| 3 | `03_extract_embeddings.ipynb` | T4 o CPU | ~5 min | 6 archivos `.npy` → Drive |
| 4 | `04_classical_models.ipynb` | CPU | ~10-15 min | 5 JSON + `classical_models.pkl` |
| 5 | `05_reduction_and_similarity.ipynb` | CPU | ~10 min | figuras UMAP + reevaluación |
| 6 | `06_final_evaluation.ipynb` | CPU | ~1 min | tabla comparativa + LaTeX |

**Total: ~1 hora** la primera vez (incluye descarga del dataset).

Para cada notebook:
1. Abrir en Colab desde GitHub
2. **Runtime → Change runtime type** según la tabla
3. **Runtime → Run all**
4. La primera celda pide autorizar Drive y (NB01) subir `kaggle.json`
5. La última celda hace `git push` automático con outputs

### Verificación final

- En GitHub: los 6 `notebooks/*.ipynb` muestran outputs al abrirlos en el browser.
- `artifacts/figures/` tiene ~15 PNG; `artifacts/metrics/` tiene ~7 JSON + 1 CSV.
- En Drive: `checkpoints/` ~75 MB, `embeddings/` ~375 MB.

### Reanudar tras desconexión de Colab

Si Colab desconecta a mitad de un notebook:
1. Vuelve a abrir el mismo notebook en Colab
2. **Runtime → Run all** → el setup detecta que el repo/Drive ya existen y solo hace `git pull` + remonta Drive
3. Los artefactos ya generados (en Drive) se reutilizan sin reentrenar

---

## 1. Setup inicial

### Opción A: pip + venv (recomendado en Windows)
```powershell
git clone https://github.com/TU_USUARIO/Malaria-Dectetion-Deeplearning.git
cd Malaria-Dectetion-Deeplearning

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

### Opción B: conda
```powershell
conda env create -f environment.yml
conda activate malaria-detection
```

### Verificar instalación
```powershell
python -c "import torch, torchvision, sklearn, umap; print('OK')"
pytest tests/test_pipeline_smoke.py -v   # debe pasar en <2 min
```

---

## 2. Flujo completo local (CPU)

```powershell
# Paso 0: Preparar datos
python -m scripts.prepare_data --config configs/data.yaml

# Paso 1: EDA (notebook — opcional)
jupyter lab notebooks/01_eda.ipynb

# Paso 2: Entrenamiento contrastivo (MUY LENTO en CPU — ver sección 3)
# Solo útil para probar con pocas épocas:
python -m scripts.train_contrastive --config configs/contrastive.yaml

# Paso 3: Extraer embeddings (requiere encoder_best.pt)
python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt

# Paso 4: Entrenar modelos clásicos (rápido en CPU, ~30 min)
python -m scripts.train_models --config configs/classical.yaml

# Paso 5: Reducción de dimensión
python -m scripts.run_reduction --config configs/reduction.yaml
python -m scripts.evaluate_similarity

# Paso 6: Ver resultados
jupyter lab notebooks/06_final_evaluation.ipynb
```

---

## 3. Flujo con Colab para entrenamiento GPU

### 3.1 Preparar el dataset en Google Drive

**Opción A — Subir manualmente:**
1. En Google Drive, crear carpeta `malaria_project/`
2. Subir `cell_images/` completa (puede tardar 30-60 min con 27k imágenes)
3. En Colab: `!cp -r '/content/drive/MyDrive/malaria_project/cell_images' ./cell_images`

**Opción B — Kaggle API (más rápido):**
```python
# En Colab:
!pip install kaggle -q
# Sube tu kaggle.json primero (API key de kaggle.com/account)
from google.colab import files
files.upload()  # sube kaggle.json
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d iarunava/cell-images-for-detecting-malaria
!unzip -q cell-images-for-detecting-malaria.zip
```

### 3.2 Ejecutar notebook 02 en Colab

1. Abrir `notebooks/02_contrastive_training.ipynb` en Colab
2. **Runtime → Change runtime type → T4 GPU** (gratis)
3. Descomenta el bloque `# COLAB`:
   ```python
   !git clone https://github.com/TU_USUARIO/Malaria-Dectetion-Deeplearning.git
   %cd Malaria-Dectetion-Deeplearning
   !pip install -r requirements.txt -q
   ```
4. Runtime → Run all
5. Esperar ~15-30 min con T4 GPU para 30 épocas

### 3.3 Descargar el checkpoint

**Opción A — Descarga directa:**
```python
from google.colab import files
files.download('artifacts/checkpoints/encoder_best.pt')
```

**Opción B — Guardar en Drive (persistente):**
```python
import shutil
shutil.copy('artifacts/checkpoints/encoder_best.pt',
            '/content/drive/MyDrive/malaria_project/encoder_best.pt')
```

### 3.4 Continuar localmente

```powershell
# Copiar encoder_best.pt a artifacts/checkpoints/ local
# Luego continuar desde paso 3:
python -m scripts.extract_embeddings --checkpoint artifacts/checkpoints/encoder_best.pt
python -m scripts.train_models --config configs/classical.yaml
```

---

## 4. Sincronización Colab ↔ VSCode / GitHub

El requisito de tener **notebooks con celdas ejecutadas en GitHub** se logra así:

### Flujo recomendado

```
VSCode (código fuente) ──push──► GitHub ──clone──► Colab (ejecuta)
                                              │
                        ◄──commit──── Colab (con outputs) ──push──►
```

### Paso a paso

#### 4.1 Conectar Colab con tu repositorio GitHub

En la primera celda del notebook en Colab:
```python
# Autenticación con GitHub en Colab
from google.colab import userdata
import subprocess

# Opción A: Token de acceso personal
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')  # guarda el token en Colab secrets
REPO_URL = f'https://{GITHUB_TOKEN}@github.com/TU_USUARIO/Malaria-Dectetion-Deeplearning.git'
!git clone {REPO_URL}
%cd Malaria-Dectetion-Deeplearning
!git config user.email "tu@email.com"
!git config user.name "Tu Nombre"
```

#### 4.2 Cómo guardar el token en Colab Secrets (seguro)
1. En Colab: Herramientas → Secrets
2. Agregar secreto `GITHUB_TOKEN` con tu Personal Access Token de GitHub
3. El token se accede con `userdata.get('GITHUB_TOKEN')` — nunca se imprime

#### 4.3 Ejecutar el notebook y subir con outputs a GitHub

Al final de cada notebook, agregar una celda:
```python
# CELDA FINAL — ejecutar después de correr todo el notebook
import subprocess

# Guardar el notebook con outputs
# (en Colab los outputs ya están en el .ipynb al guardar)
!jupyter nbconvert --to notebook --inplace notebooks/01_eda.ipynb 2>/dev/null || true

# Commit y push con outputs
!git add notebooks/01_eda.ipynb artifacts/figures/ artifacts/metrics/
!git commit -m "nb01: EDA ejecutado con outputs y figuras"
!git push origin main
```

#### 4.4 Plantilla de celda de sincronización (copiar a cada notebook)

```python
# ═══════════════════════════════════════════════════════════
# SYNC CON GITHUB — ejecutar al final del notebook
# ═══════════════════════════════════════════════════════════
import subprocess
from pathlib import Path

NOTEBOOK_NAME = "01_eda"  # cambiar por el nombre del notebook

# Agregar archivos relevantes
files_to_add = [
    f"notebooks/{NOTEBOOK_NAME}.ipynb",
    "artifacts/figures/",
    "artifacts/metrics/",
    "data/processed/",   # splits CSV (pequeños)
]

for f in files_to_add:
    if Path(f).exists():
        subprocess.run(["git", "add", f])

result = subprocess.run(
    ["git", "commit", "-m", f"nb{NOTEBOOK_NAME}: ejecutado con outputs"],
    capture_output=True, text=True
)
print(result.stdout or result.stderr)

push = subprocess.run(["git", "push", "origin", "main"], capture_output=True, text=True)
print(push.stdout or push.stderr)
print("✓ Sincronizado con GitHub")
```

---

## 5. Ejecutar notebooks en GitHub con outputs

### Estrategia recomendada

| Archivo | ¿Commitear? | Contenido |
|---|---|---|
| `notebooks/*.ipynb` | **SÍ** — con outputs | El profesor verá los resultados |
| `data/processed/*.csv` | **SÍ** | Splits reproducibles |
| `artifacts/figures/*.png` | **SÍ** | Figuras para el reporte |
| `artifacts/metrics/*.json` | **SÍ** | Métricas completas |
| `artifacts/checkpoints/*.pt` | **NO** (>100MB) | Demasiado pesado para git |
| `data/embeddings/*.npy` | **NO** (~500MB) | Demasiado pesado |
| `cell_images/` | **NO** | Demasiado pesado |

### Flujo para entregar el repositorio completo con outputs

```powershell
# 1. VSCode local: Asegúrate de que el código esté actualizado
git status
git add src/ scripts/ tests/ configs/ notebooks/ docs/ requirements.txt README.md .gitignore
git commit -m "proyecto: estructura y código completos"
git push origin main

# 2. Colab: Ejecuta cada notebook y sincroniza (ver sección 4.4)
# Notebook 01 → commit con outputs
# Notebook 02 → commit (el entrenamiento puede no caber, pero las curvas sí)
# Notebook 03 → commit con outputs
# Notebook 04 → commit con outputs + tablas
# Notebook 05 → commit con outputs + figuras UMAP
# Notebook 06 → commit con tabla final + LaTeX

# 3. Verificar en GitHub que los notebooks tienen outputs visibles
```

### Alternativa: nbconvert para limpiar outputs antes de commitear (solo código)

Si prefieres commitear notebooks **SIN** outputs (solo código), luego ejecutarlos:
```powershell
# Limpiar outputs de todos los notebooks
jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
git add notebooks/
git commit -m "notebooks: limpios (sin outputs)"
```

---

## 6. Solución de problemas comunes

### Error: `num_workers > 0` en Windows
```
# En Windows, usar num_workers=0 para DataLoader
# El config ya lo maneja automáticamente (scripts/train_contrastive.py)
# Si sigues viendo el error, edita configs/contrastive.yaml:
#   num_workers: 0
```

### Error: CUDA out of memory en Colab
```
# Reducir batch_size en configs/contrastive.yaml:
#   batch_size_gpu: 128  (en lugar de 256)
# O reducir img_size a 64
```

### Error: Colab desconectado durante entrenamiento
```python
# El entrenamiento guarda checkpoint cada epoch en encoder_last.pt
# Para resumir desde el último checkpoint:
# Modifica scripts/train_contrastive.py para cargar encoder_last.pt si existe
# O simplemente ajusta los epochs restantes en configs/contrastive.yaml
```

### Error: `ModuleNotFoundError: No module named 'src'`
```powershell
# Asegúrate de ejecutar desde la raíz del repo:
cd Malaria-Dectetion-Deeplearning
python -m scripts.prepare_data  # con python -m, no python scripts/...
```

### Error: `umap not installed`
```powershell
pip install umap-learn
```

### Warning: `UserWarning: CUDA initialization` en CPU
```
# Normal si no tienes GPU. El código detecta automáticamente CPU vs GPU.
```

### Tests fallan por timeout
```powershell
# Smoke test debe terminar en <2 min. Si tarda más:
pytest tests/test_pipeline_smoke.py -v --timeout=300
```
