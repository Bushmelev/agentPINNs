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
from .losses import LossEvaluator, LossPack
from .models import MLP
from .optim import make_lbfgs_optimizer, make_optimizer, make_scheduler
from .settings import set_seed


@dataclass
class TrainResult:
    model: nn.Module
    history: dict[str, Any]
    controller: WeightController
    elapsed_sec: float


@dataclass
class OptimizerPhase:
    name: str
    steps: int
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None


def _empty_history(component_names: list[str], controller_name: str) -> dict[str, Any]:
    return {
        "controller": controller_name,
        "component_names": component_names,
        "equal_weight_total": [],
        "weighted_total": [],
        "components": {name: [] for name in component_names},
        "weights": [],
        "lr": [],
        "optimizer_phase": [],
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


def _normalize_optimizer_mode(mode: str) -> str:
    value = mode.lower().replace("-", "_").replace("+", "_")
    value = value.replace("__", "_")
    aliases = {
        "only_adam": "adam",
        "adam_lbfgs": "adam_lbfgs",
        "adam_l_bfgs": "adam_lbfgs",
        "adam_then_lbfgs": "adam_lbfgs",
        "only_lbfgs": "lbfgs",
        "l_bfgs": "lbfgs",
    }
    value = aliases.get(value, value)
    if value not in {"adam", "adam_lbfgs", "lbfgs"}:
        raise ValueError(
            "optimizer_mode must be one of: adam, adam_lbfgs, lbfgs"
        )
    return value


def _phase_step_counts(train_cfg: TrainingConfig) -> tuple[int, int]:
    mode = _normalize_optimizer_mode(train_cfg.optimizer_mode)
    total_steps = int(train_cfg.steps)
    if total_steps <= 0:
        raise ValueError("training.steps must be positive")

    adam_steps = train_cfg.adam_steps
    lbfgs_steps = train_cfg.lbfgs_steps

    if mode == "adam":
        return int(adam_steps if adam_steps is not None else total_steps), 0
    if mode == "lbfgs":
        return 0, int(lbfgs_steps if lbfgs_steps is not None else total_steps)

    if adam_steps is None and lbfgs_steps is None:
        resolved_adam = max(1, int(round(0.8 * total_steps)))
        resolved_lbfgs = max(1, total_steps - resolved_adam)
        return resolved_adam, resolved_lbfgs
    if adam_steps is None:
        resolved_lbfgs = int(lbfgs_steps or 0)
        return max(total_steps - resolved_lbfgs, 0), resolved_lbfgs
    if lbfgs_steps is None:
        resolved_adam = int(adam_steps)
        return resolved_adam, max(total_steps - resolved_adam, 0)
    return int(adam_steps), int(lbfgs_steps)


def _make_optimizer_phases(
    params: list[torch.nn.Parameter],
    train_cfg: TrainingConfig,
) -> list[OptimizerPhase]:
    adam_steps, lbfgs_steps = _phase_step_counts(train_cfg)
    phases: list[OptimizerPhase] = []
    if adam_steps > 0:
        optimizer = make_optimizer(
            params,
            train_cfg.optimizer,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        scheduler = make_scheduler(
            optimizer,
            train_cfg.scheduler,
            **train_cfg.scheduler_kwargs,
        )
        phases.append(OptimizerPhase("adam", adam_steps, optimizer, scheduler))
    if lbfgs_steps > 0:
        optimizer = make_lbfgs_optimizer(
            params,
            lr=train_cfg.lbfgs_lr,
            max_iter=train_cfg.lbfgs_max_iter,
            max_eval=train_cfg.lbfgs_max_eval,
            history_size=train_cfg.lbfgs_history_size,
            tolerance_grad=train_cfg.lbfgs_tolerance_grad,
            tolerance_change=train_cfg.lbfgs_tolerance_change,
            line_search_fn=train_cfg.lbfgs_line_search_fn,
        )
        phases.append(OptimizerPhase("lbfgs", lbfgs_steps, optimizer))
    if not phases:
        raise ValueError("optimizer schedule has zero steps")
    return phases


def _history_values(
    loss_pack: LossPack,
    component_names: list[str],
    weights_t: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    raw_losses = np.array(
        [
            float(loss_pack.by_name[name].detach().cpu().item())
            for name in component_names
        ],
        dtype=np.float64,
    )
    weights_np = weights_t.detach().cpu().numpy().astype(np.float32)
    equal_total = float(np.mean(raw_losses))
    weighted_total = float(np.sum(weights_np * raw_losses))
    return raw_losses, weights_np, equal_total, weighted_total


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
    phases = _make_optimizer_phases(optimizer_params, train_cfg)
    total_steps = sum(phase.steps for phase in phases)
    history = _empty_history(component_names, controller.name)
    start_time = time.time()
    step = 0

    for phase in phases:
        for _ in range(phase.steps):
            step += 1
            batches = loss_evaluator.draw_batches()
            if phase.name == "lbfgs":
                loss_pack = loss_evaluator.compute(train_model, batches)
                objective, weights_t = controller.objective(
                    loss_pack.values,
                    model,
                    step,
                    update_state=True,
                )
                raw_losses, weights_np, equal_total, weighted_total = _history_values(
                    loss_pack,
                    component_names,
                    weights_t,
                )
                del objective, loss_pack, weights_t

                def closure() -> torch.Tensor:
                    phase.optimizer.zero_grad(set_to_none=True)
                    closure_pack = loss_evaluator.compute(train_model, batches)
                    closure_objective, _ = controller.objective(
                        closure_pack.values,
                        model,
                        step,
                        update_state=False,
                    )
                    closure_objective.backward()
                    return closure_objective

                phase.optimizer.step(closure)
            else:
                phase.optimizer.zero_grad(set_to_none=True)
                loss_pack = loss_evaluator.compute(train_model, batches)
                objective, weights_t = controller.objective(
                    loss_pack.values,
                    model,
                    step,
                    update_state=True,
                )
                objective.backward()
                phase.optimizer.step()
                if phase.scheduler is not None:
                    phase.scheduler.step()
                raw_losses, weights_np, equal_total, weighted_total = _history_values(
                    loss_pack,
                    component_names,
                    weights_t,
                )
            progress = step / float(max(total_steps, 1))
            snapshot = StepSnapshot(
                step=step,
                total=equal_total,
                losses=raw_losses,
                weights=weights_np,
                progress=progress,
                done=step == total_steps,
            )
            extras = controller.after_step(snapshot, baseline_history)

            history["equal_weight_total"].append(equal_total)
            history["weighted_total"].append(weighted_total)
            for name, value in zip(component_names, raw_losses):
                history["components"][name].append(float(value))
            history["weights"].append(weights_np.tolist())
            history["lr"].append(float(phase.optimizer.param_groups[0]["lr"]))
            history["optimizer_phase"].append(phase.name)
            history["agent_reward"].append(extras.get("agent_reward"))

            if train_cfg.log_every > 0 and step % train_cfg.log_every == 0:
                pieces = " ".join(
                    f"{name}={history['components'][name][-1]:.3e}"
                    for name in component_names
                )
                print(
                    f"[{spec.name}/{controller.name}] phase={phase.name} "
                    f"step={step}/{total_steps} equal={equal_total:.3e} "
                    f"weighted={weighted_total:.3e} {pieces}"
                )

    return TrainResult(
        model=model,
        history=history,
        controller=controller,
        elapsed_sec=time.time() - start_time,
    )
