"""
Verifica la integridad del dataset y genera los splits estratificados fijos.

Uso:
    python -m scripts.prepare_data --config configs/data.yaml
"""
import argparse
import sys
from pathlib import Path

# Asegurar que la raíz del repo está en PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.split import make_stratified_split
from src.utils.io import load_config
from src.utils.logging import get_logger
from src.utils.seed import set_global_seed

logger = get_logger("prepare_data")


def verify_images(dataset_root: Path, class_map: dict[str, int]) -> int:
    """Verifica que las imágenes se pueden abrir; retorna número de corruptas."""
    from PIL import Image, UnidentifiedImageError

    corrupted = 0
    total = 0
    for class_name in class_map:
        class_dir = dataset_root / class_name
        imgs = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        for p in imgs:
            total += 1
            try:
                with Image.open(p) as img:
                    img.verify()
            except (UnidentifiedImageError, Exception):
                logger.warning(f"Imagen corrupta: {p}")
                corrupted += 1
    logger.info(f"Verificadas {total} imágenes — {corrupted} corruptas")
    return corrupted


def main(cfg_path: str) -> None:
    cfg = load_config(cfg_path)
    set_global_seed(cfg["seed"])

    dataset_root = Path(cfg["dataset_root"])
    processed_dir = Path(cfg["processed_dir"])
    class_map = cfg.get("classes", {"Parasitized": 1, "Uninfected": 0})

    logger.info(f"Dataset root: {dataset_root.resolve()}")
    assert dataset_root.exists(), f"No existe: {dataset_root.resolve()}"

    # Conteo por clase
    for class_name in class_map:
        class_dir = dataset_root / class_name
        n = len(list(class_dir.glob("*.png"))) + len(list(class_dir.glob("*.jpg")))
        logger.info(f"  {class_name}: {n} imágenes")

    # Verificar integridad
    corrupted = verify_images(dataset_root, class_map)
    if corrupted > 0:
        logger.warning(f"{corrupted} imágenes corruptas detectadas — se excluirán del split")

    # Generar splits
    split_cfg = cfg.get("split", {"train": 0.70, "val": 0.15, "test": 0.15})
    train_df, val_df, test_df = make_stratified_split(
        dataset_root=dataset_root,
        processed_dir=processed_dir,
        train_frac=split_cfg["train"],
        val_frac=split_cfg["val"],
        seed=cfg["seed"],
        class_map=class_map,
    )

    logger.info("Split estratificado generado:")
    logger.info(f"  Train: {len(train_df)} ({train_df['label'].mean():.3f} pos)")
    logger.info(f"  Val:   {len(val_df)} ({val_df['label'].mean():.3f} pos)")
    logger.info(f"  Test:  {len(test_df)} ({test_df['label'].mean():.3f} pos)")
    logger.info(f"CSVs guardados en {processed_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    main(args.config)
