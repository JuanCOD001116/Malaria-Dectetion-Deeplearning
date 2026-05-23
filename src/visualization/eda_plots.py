"""Plots para EDA: ejemplos, histogramas, brillo/contraste."""
from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_class_examples(
    df,
    n_per_class: int = 8,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Muestra n_per_class imágenes por clase."""
    random.seed(seed)
    class_names = sorted(df["class_name"].unique())
    fig, axes = plt.subplots(len(class_names), n_per_class, figsize=(2 * n_per_class, 3 * len(class_names)))

    for row, cls in enumerate(class_names):
        subset = df[df["class_name"] == cls]["path"].tolist()
        samples = random.sample(subset, min(n_per_class, len(subset)))
        for col, path in enumerate(samples):
            img = np.array(Image.open(path).convert("RGB"))
            ax = axes[row, col] if len(class_names) > 1 else axes[col]
            ax.imshow(img)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(cls, fontsize=11, fontweight="bold", rotation=0, labelpad=60)

    fig.suptitle("Ejemplos aleatorios por clase", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_size_distribution(df, save_path: str | Path | None = None) -> plt.Figure:
    """Distribución de tamaños de imagen."""
    widths, heights = [], []
    for path in df["path"]:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(widths, bins=30, color="steelblue", alpha=0.8)
    axes[0].set_xlabel("Ancho (px)")
    axes[0].set_ylabel("Frecuencia")
    axes[0].set_title(f"Distribución de anchos (median={np.median(widths):.0f}px)")

    axes[1].hist(heights, bins=30, color="salmon", alpha=0.8)
    axes[1].set_xlabel("Alto (px)")
    axes[1].set_ylabel("Frecuencia")
    axes[1].set_title(f"Distribución de altos (median={np.median(heights):.0f}px)")

    fig.suptitle("Distribución de tamaños de imagen", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_intensity_histograms(
    df,
    n_samples: int = 200,
    seed: int = 42,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Histogramas de intensidad de píxeles por canal y por clase."""
    rng = np.random.default_rng(seed)
    class_names = sorted(df["class_name"].unique())
    colors_per_class = {"Parasitized": "tomato", "Uninfected": "steelblue"}
    channels = ["Rojo", "Verde", "Azul"]

    fig, axes = plt.subplots(len(class_names), 3, figsize=(14, 4 * len(class_names)))

    for row, cls in enumerate(class_names):
        paths = df[df["class_name"] == cls]["path"].tolist()
        selected = rng.choice(paths, min(n_samples, len(paths)), replace=False).tolist()

        all_pixels = {0: [], 1: [], 2: []}
        for p in selected:
            try:
                arr = np.array(Image.open(p).convert("RGB"))
                for c in range(3):
                    all_pixels[c].extend(arr[:, :, c].flatten().tolist())
            except Exception:
                pass

        color = colors_per_class.get(cls, "gray")
        for col, ch_name in enumerate(channels):
            ax = axes[row, col] if len(class_names) > 1 else axes[col]
            ax.hist(all_pixels[col], bins=50, color=color, alpha=0.7, density=True)
            ax.set_xlabel("Intensidad [0-255]")
            ax.set_ylabel("Densidad")
            ax.set_title(f"{cls} — Canal {ch_name}")

    fig.suptitle("Histogramas de intensidad por canal y clase", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_augmentation_examples(
    sample_path: str,
    transform,
    n_augmentations: int = 6,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Muestra la imagen original y n_augmentations augmentadas."""
    img_pil = Image.open(sample_path).convert("RGB")
    img_np = np.array(img_pil)

    fig, axes = plt.subplots(1, n_augmentations + 1, figsize=(2.5 * (n_augmentations + 1), 3))
    axes[0].imshow(img_np)
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])

    for i in range(1, n_augmentations + 1):
        aug = transform(img_pil).numpy().transpose(1, 2, 0)
        aug = np.clip(aug * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        axes[i].imshow(aug)
        axes[i].set_title(f"Aug {i}", fontsize=9)
        axes[i].axis("off")

    fig.suptitle("Ejemplos de augmentations contrastivas", fontweight="bold")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
