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


@dataclass
class RelativeL2Metric:
    xt: torch.Tensor
    target: torch.Tensor
    denominator: torch.Tensor
    chunk_size: int

    def __call__(self, model: nn.Module) -> float:
        squared_error = torch.zeros((), dtype=torch.float32, device=self.xt.device)
        with torch.no_grad():
            for start in range(0, self.xt.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, self.xt.shape[0])
                prediction = model(self.xt[start:stop]).reshape(-1)
                diff = prediction - self.target[start:stop]
                squared_error = squared_error + torch.sum(diff * diff)
        return float(torch.sqrt(squared_error / self.denominator).detach().cpu().item())


def _empty_history(component_names: list[str], controller_name: str) -> dict[str, Any]:
    return {
        "controller": controller_name,
        "component_names": component_names,
        "equal_weight_total": [],
        "weighted_total": [],
        "relative_l2": [],
        "components": {name: [] for name in component_names},
        "weights": [],
        "lr": [],
        "progress": [],
        "agent_progress": [],
        "optimizer_phase": [],
        "weights_frozen": [],
        "agent_reward": [],
        "agent_sigma": [],
        "agent_frozen": [],
        "lbfgs_closure_calls": [],
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
    model_params: list[torch.nn.Parameter],
    controller_params: list[torch.nn.Parameter],
    train_cfg: TrainingConfig,
    controller: WeightController,
) -> list[OptimizerPhase]:
    adam_steps, lbfgs_steps = _phase_step_counts(train_cfg)
    all_params = [*model_params, *controller_params]
    phases: list[OptimizerPhase] = []
    if adam_steps > 0:
        optimizer = make_optimizer(
            all_params,
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
        lbfgs_params = (
            model_params
            if _weights_frozen_in_phase("lbfgs", train_cfg, controller)
            else all_params
        )
        optimizer = make_lbfgs_optimizer(
            lbfgs_params,
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


def _weights_frozen_in_phase(
    phase_name: str,
    train_cfg: TrainingConfig,
    controller: WeightController,
) -> bool:
    if phase_name != "lbfgs":
        return False
    if _lbfgs_weight_mode(train_cfg) == "equal":
        return True
    if train_cfg.freeze_weights_during_lbfgs and controller.name != "fixed":
        return True
    return train_cfg.freeze_agent_during_lbfgs and controller.uses_agent


def _lbfgs_weight_mode(train_cfg: TrainingConfig) -> str:
    value = train_cfg.lbfgs_weight_mode.lower().replace("-", "_")
    aliases = {
        "controller": "controller",
        "learned": "controller",
        "agent": "controller",
        "equal": "equal",
        "uniform": "equal",
        "fixed": "equal",
    }
    value = aliases.get(value, value)
    if value not in {"controller", "equal"}:
        raise ValueError("training.lbfgs_weight_mode must be 'controller' or 'equal'")
    return value


def _agent_active_in_phase(
    phase: OptimizerPhase,
    train_cfg: TrainingConfig,
    controller: WeightController,
) -> bool:
    return not (
        controller.uses_agent
        and _weights_frozen_in_phase(phase.name, train_cfg, controller)
    )


def _build_relative_l2_metric(
    spec: EquationSpec,
    device: torch.device,
    chunk_size: int,
) -> RelativeL2Metric | None:
    if spec.reference_solver is None:
        return None
    x, t, u = spec.solve_reference()
    expected_shape = (len(x), len(t))
    if u.shape == (len(t), len(x)):
        u = u.T
    if u.shape != expected_shape:
        raise ValueError(
            f"Reference solution shape must be {expected_shape}, got {u.shape}"
        )
    grid_x, grid_t = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([grid_x.reshape(-1), grid_t.reshape(-1)], axis=1)
    target_np = u.reshape(-1).astype(np.float32)
    target = torch.tensor(target_np, dtype=torch.float32, device=device)
    denominator = torch.clamp(torch.sum(target * target), min=1e-12)
    return RelativeL2Metric(
        xt=torch.tensor(xt, dtype=torch.float32, device=device),
        target=target,
        denominator=denominator,
        chunk_size=max(1, int(chunk_size)),
    )


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


def _equal_weight_objective(losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.full_like(losses, 1.0 / max(int(losses.numel()), 1))
    return torch.sum(weights * losses), weights.detach()


def _lbfgs_objective(
    *,
    losses: torch.Tensor,
    model: nn.Module,
    step: int,
    train_cfg: TrainingConfig,
    controller: WeightController,
    weights_frozen: bool,
    update_state: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _lbfgs_weight_mode(train_cfg) == "equal":
        return _equal_weight_objective(losses)
    if weights_frozen:
        return controller.frozen_objective(losses, model, step)
    return controller.objective(losses, model, step, update_state=update_state)


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
        full_batch=train_cfg.full_batch,
    )
    component_names = loss_evaluator.component_names
    model = MLP(model_cfg.layers, activation=model_cfg.activation).to(device)
    train_model = _maybe_compile(model, train_cfg.compile_model)
    controller.bind(component_names, np.ones(len(component_names), dtype=np.float32), device)

    model_params = list(model.parameters())
    controller_params = list(controller.parameters()) if controller.trainable else []
    phases = _make_optimizer_phases(
        model_params,
        controller_params,
        train_cfg,
        controller,
    )
    total_steps = sum(phase.steps for phase in phases)
    agent_total_steps = sum(
        phase.steps
        for phase in phases
        if _agent_active_in_phase(phase, train_cfg, controller)
    )
    relative_l2_metric = (
        _build_relative_l2_metric(
            spec,
            device,
            train_cfg.relative_l2_chunk_size,
        )
        if train_cfg.relative_l2_every > 0
        else None
    )
    history = _empty_history(component_names, controller.name)
    history["batch_info"] = loss_evaluator.batch_info
    start_time = time.time()
    step = 0
    agent_step = 0

    for phase in phases:
        for _ in range(phase.steps):
            step += 1
            lbfgs_closure_calls = None
            weights_frozen = _weights_frozen_in_phase(
                phase.name,
                train_cfg,
                controller,
            )
            agent_active = _agent_active_in_phase(phase, train_cfg, controller)
            if agent_active:
                agent_step += 1
            batches = loss_evaluator.draw_batches()
            if phase.name == "lbfgs":
                loss_pack = loss_evaluator.compute(train_model, batches)
                objective, weights_t = _lbfgs_objective(
                    losses=loss_pack.values,
                    model=model,
                    step=step,
                    train_cfg=train_cfg,
                    controller=controller,
                    weights_frozen=weights_frozen,
                    update_state=True,
                )
                raw_losses, weights_np, equal_total, weighted_total = _history_values(
                    loss_pack,
                    component_names,
                    weights_t,
                )
                del objective, loss_pack, weights_t

                lbfgs_closure_calls = 0

                def closure() -> torch.Tensor:
                    nonlocal lbfgs_closure_calls
                    lbfgs_closure_calls += 1
                    phase.optimizer.zero_grad(set_to_none=True)
                    closure_pack = loss_evaluator.compute(train_model, batches)
                    closure_objective, _ = _lbfgs_objective(
                        losses=closure_pack.values,
                        model=model,
                        step=step,
                        train_cfg=train_cfg,
                        controller=controller,
                        weights_frozen=weights_frozen,
                        update_state=False,
                    )
                    closure_objective.backward()
                    return closure_objective

                phase.optimizer.step(closure)
                loss_pack = loss_evaluator.compute(train_model, batches)
                objective, weights_t = _lbfgs_objective(
                    losses=loss_pack.values,
                    model=model,
                    step=step,
                    train_cfg=train_cfg,
                    controller=controller,
                    weights_frozen=weights_frozen,
                    update_state=False,
                )
                raw_losses, weights_np, equal_total, weighted_total = _history_values(
                    loss_pack,
                    component_names,
                    weights_t,
                )
                del objective, loss_pack, weights_t
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
            agent_progress = agent_step / float(max(agent_total_steps, 1))
            relative_l2 = None
            if relative_l2_metric is not None and (
                step % train_cfg.relative_l2_every == 0 or step == total_steps
            ):
                relative_l2 = relative_l2_metric(train_model)
            snapshot = StepSnapshot(
                step=step,
                total=equal_total,
                losses=raw_losses,
                weights=weights_np,
                relative_l2=relative_l2,
                progress=progress,
                agent_progress=agent_progress,
                done=step == total_steps,
            )
            agent_frozen = (
                controller.uses_agent
                and (
                    not agent_active
                    or not getattr(getattr(controller, "agent", None), "trainable", True)
                )
            )
            if agent_frozen:
                if agent_active:
                    extras = controller.after_step(snapshot, baseline_history)
                else:
                    extras = controller.frozen_step_extras()
            else:
                extras = controller.after_step(snapshot, baseline_history)

            history["equal_weight_total"].append(equal_total)
            history["weighted_total"].append(weighted_total)
            history["relative_l2"].append(relative_l2)
            for name, value in zip(component_names, raw_losses):
                history["components"][name].append(float(value))
            history["weights"].append(weights_np.tolist())
            history["lr"].append(float(phase.optimizer.param_groups[0]["lr"]))
            history["progress"].append(progress)
            history["agent_progress"].append(agent_progress)
            history["optimizer_phase"].append(phase.name)
            history["weights_frozen"].append(weights_frozen)
            history["agent_reward"].append(extras.get("agent_reward"))
            history["agent_sigma"].append(extras.get("agent_sigma"))
            history["agent_frozen"].append(agent_frozen)
            history["lbfgs_closure_calls"].append(lbfgs_closure_calls)

            if train_cfg.log_every > 0 and step % train_cfg.log_every == 0:
                pieces = " ".join(
                    f"{name}={history['components'][name][-1]:.3e}"
                    for name in component_names
                )
                print(
                    f"[{spec.name}/{controller.name}] phase={phase.name} "
                    f"step={step}/{total_steps} equal={equal_total:.3e} "
                    f"weighted={weighted_total:.3e} "
                    f"rel_l2={relative_l2 if relative_l2 is not None else float('nan'):.3e} "
                    f"{pieces}"
                )

    return TrainResult(
        model=model,
        history=history,
        controller=controller,
        elapsed_sec=time.time() - start_time,
    )
