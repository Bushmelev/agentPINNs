from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(name: str = "auto") -> torch.device:
    value = name.lower()
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_torch() -> None:
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
