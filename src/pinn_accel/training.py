from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import ModelConfig, TrainingConfig
from .controllers import StepSnapshot, WeightController
from .equations.base import EquationSpec
from .losses import LossEvaluator
from .models import MLP
from .optim import make_optimizer, make_scheduler
from .settings import set_seed


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, Any]
    controller: WeightController
    elapsed_sec: float


def _empty_history(component_names: list[str], controller_name: str) -> dict[str, Any]:
    return {
        "controller": controller_name,
        "component_names": component_names,
        "equal_weight_total": [],
        "weighted_total": [],
        "components": {name: [] for name in component_names},
        "weights": [],
        "lr": [],
        "agent_reward": [],
    }


def _maybe_compile(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    try:
        return torch.compile(model)
    except Exception as exc:
        print(f"torch.compile disabled: {exc}")
        return model


def train_one(
    *,
    spec: EquationSpec,
    model_cfg: ModelConfig,
    train_cfg: TrainingConfig,
    controller: WeightController,
    device: torch.device,
    seed: int,
    baseline_history: dict[str, Any] | None = None,
) -> TrainResult:
    set_seed(seed)
    loss_evaluator = LossEvaluator(
        spec,
        batch_sizes=train_cfg.batch_sizes,
        pool_sizes=train_cfg.pool_sizes,
        device=device,
        seed=seed,
    )
    component_names = loss_evaluator.component_names
    model = MLP(model_cfg.layers, activation=model_cfg.activation).to(device)
    train_model = _maybe_compile(model, train_cfg.compile_model)
    controller.bind(component_names, np.ones(len(component_names), dtype=np.float32), device)

    optimizer_params = list(model.parameters())
    if controller.trainable:
        optimizer_params.extend(list(controller.parameters()))
    optimizer = make_optimizer(
        optimizer_params,
        train_cfg.optimizer,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = make_scheduler(optimizer, train_cfg.scheduler, **train_cfg.scheduler_kwargs)
    history = _empty_history(component_names, controller.name)
    start_time = time.time()

    for step in range(1, train_cfg.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_pack = loss_evaluator.compute(train_model)
        objective, weights_t = controller.objective(loss_pack.values, model, step)
        objective.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        raw_losses = np.array(
            [float(loss_pack.by_name[name].detach().cpu().item()) for name in component_names],
            dtype=np.float64,
        )
        weights_np = weights_t.detach().cpu().numpy().astype(np.float32)
        equal_total = float(np.mean(raw_losses))
        weighted_total = float(np.sum(weights_np * raw_losses))
        progress = step / float(max(train_cfg.steps, 1))
        snapshot = StepSnapshot(
            step=step,
            total=equal_total,
            losses=raw_losses,
            weights=weights_np,
            progress=progress,
            done=step == train_cfg.steps,
        )
        extras = controller.after_step(snapshot, baseline_history)

        history["equal_weight_total"].append(equal_total)
        history["weighted_total"].append(weighted_total)
        for name, value in zip(component_names, raw_losses):
            history["components"][name].append(float(value))
        history["weights"].append(weights_np.tolist())
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["agent_reward"].append(extras.get("agent_reward"))

        if train_cfg.log_every > 0 and step % train_cfg.log_every == 0:
            pieces = " ".join(
                f"{name}={history['components'][name][-1]:.3e}" for name in component_names
            )
            print(
                f"[{spec.name}/{controller.name}] "
                f"step={step}/{train_cfg.steps} equal={equal_total:.3e} "
                f"weighted={weighted_total:.3e} {pieces}"
            )

    return TrainResult(
        model=model,
        history=history,
        controller=controller,
        elapsed_sec=time.time() - start_time,
    )
