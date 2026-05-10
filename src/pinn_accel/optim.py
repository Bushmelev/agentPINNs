from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def make_optimizer(
    params: Iterable[torch.nn.Parameter],
    name: str,
    *,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    value = name.lower()
    if value == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if value == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if value == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unknown optimizer: {name}")


def make_lbfgs_optimizer(
    params: Iterable[torch.nn.Parameter],
    *,
    lr: float = 1.0,
    max_iter: int = 20,
    max_eval: int | None = None,
    history_size: int = 100,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    line_search_fn: str | None = "strong_wolfe",
) -> torch.optim.LBFGS:
    return torch.optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        history_size=history_size,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn=line_search_fn,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    **kwargs: Any,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    value = name.lower()
    if value in {"none", "constant"}:
        return None
    if value == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(kwargs.get("step_size", 500)),
            gamma=float(kwargs.get("gamma", 0.5)),
        )
    if value == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(kwargs.get("t_max", 1000)),
            eta_min=float(kwargs.get("eta_min", 0.0)),
        )
    raise ValueError(f"Unknown scheduler: {name}")
