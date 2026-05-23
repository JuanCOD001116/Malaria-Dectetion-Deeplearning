"""Reproducibilidad global: fija todas las semillas conocidas."""
import os
import random
import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Fija semillas en Python, NumPy, PyTorch (CPU y GPU) y variables de entorno."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Determinismo en cuDNN — puede reducir rendimiento, pero garantiza reproducibilidad
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
