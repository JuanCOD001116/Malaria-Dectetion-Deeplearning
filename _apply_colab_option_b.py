"""Aplica la Opción B (sin auto-push) a los 6 notebooks.

Reemplaza:
  - Celda SETUP: clone público del repo del compañero, sin token
  - Celda SYNC: files.download() para bajar el .ipynb ejecutado al PC

Mantiene las celdas KAGGLE (NB01-03) intactas porque siguen referenciando DRIVE_ROOT.

Uso: python _apply_colab_option_b.py
"""
import json
from pathlib import Path

NEW_SETUP = '''# ════════════════════════════════════════════════════════════════
# SETUP — Colab sin auto-push (clone público del repo del compañero)
# No requiere GitHub token; al final descargas el .ipynb a tu PC.
# ════════════════════════════════════════════════════════════════
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    from google.colab import drive

    REPO     = "Malaria-Dectetion-Deeplearning"
    REPO_URL = "https://github.com/JuanCOD001116/Malaria-Dectetion-Deeplearning.git"

    if not os.path.exists(f"/content/{REPO}"):
        ret = get_ipython().getoutput(f"git clone {REPO_URL}")
        print("\\n".join(ret))
        assert os.path.exists(f"/content/{REPO}"), \\
            "git clone falló — revisa el output arriba (¿el repo es público?)"

    get_ipython().run_line_magic("cd", f"/content/{REPO}")
    get_ipython().system("git pull origin main")
    get_ipython().system("pip install -r requirements.txt -q")

    drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT = "/content/drive/MyDrive/malaria_project"
    get_ipython().system(f"mkdir -p {DRIVE_ROOT}/checkpoints {DRIVE_ROOT}/embeddings")
    get_ipython().system("rm -rf artifacts/checkpoints data/embeddings")
    get_ipython().system("mkdir -p artifacts data")
    get_ipython().system(f"ln -sfn {DRIVE_ROOT}/checkpoints artifacts/checkpoints")
    get_ipython().system(f"ln -sfn {DRIVE_ROOT}/embeddings   data/embeddings")
    get_ipython().system("mkdir -p artifacts/figures artifacts/metrics artifacts/logs data/processed")
    print("✓ Colab listo (modo sin push). Pesados → Drive, ligeros → repo local.")

cwd = Path().resolve()
REPO_ROOT = cwd if (cwd / "src").exists() else cwd.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
print(f"Working dir: {REPO_ROOT}")'''


def new_sync(notebook_name: str) -> str:
    return f'''# ════════════════════════════════════════════════════════════════
# DESCARGA MANUAL — baja el .ipynb ejecutado a tu PC
# Sin auto-push: tú subes/entregas el notebook por el medio que prefieras.
# ════════════════════════════════════════════════════════════════
if IN_COLAB:
    NOTEBOOK = "{notebook_name}"
    # Forzar guardado del .ipynb (preserva outputs y figuras embebidas)
    try:
        from google.colab import _message
        _message.blocking_request("save_notebook", request="", timeout_sec=10)
    except Exception:
        pass
    # Descargar a tu PC (revisa carpeta de descargas)
    from google.colab import files
    files.download(f"notebooks/{{NOTEBOOK}}.ipynb")
    print(f"✓ Descarga iniciada: {{NOTEBOOK}}.ipynb")'''


NOTEBOOKS = {
    "notebooks/01_eda.ipynb": "01_eda",
    "notebooks/02_contrastive_training.ipynb": "02_contrastive_training",
    "notebooks/03_extract_embeddings.ipynb": "03_extract_embeddings",
    "notebooks/04_classical_models.ipynb": "04_classical_models",
    "notebooks/05_reduction_and_similarity.ipynb": "05_reduction_and_similarity",
    "notebooks/06_final_evaluation.ipynb": "06_final_evaluation",
}


def src_to_list(text: str) -> list:
    """Convierte el source a la lista de líneas terminadas en \\n que espera el formato ipynb."""
    lines = text.split("\n")
    return [ln + "\n" for ln in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def replace_cell(cell, new_source: str) -> None:
    cell["source"] = src_to_list(new_source)
    cell["outputs"] = []
    cell["execution_count"] = None


def patch_notebook(path: str, nb_name: str) -> tuple[bool, bool]:
    nb = json.loads(Path(path).read_text(encoding="utf-8"))
    setup_done = sync_done = False
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if "SETUP — Local o Colab" in src or "SETUP — Colab sin auto-push" in src:
            replace_cell(cell, NEW_SETUP)
            setup_done = True
        elif "SYNC CON GITHUB" in src or "DESCARGA MANUAL" in src:
            replace_cell(cell, new_sync(nb_name))
            sync_done = True
    Path(path).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return setup_done, sync_done


if __name__ == "__main__":
    for path, name in NOTEBOOKS.items():
        s, y = patch_notebook(path, name)
        marker = "[OK]" if (s and y) else "[??]"
        print(f"{marker} {path}: SETUP={s} SYNC={y}")
