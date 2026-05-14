from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from pinn_accel.artifacts import ArtifactStore, to_jsonable
from pinn_accel.config import ExperimentConfig, TrainingConfig
from pinn_accel.controllers import (
    StepSnapshot,
    WeightController,
    controller_needs_baseline,
    make_controller,
)
from pinn_accel.equations import get_equation
from pinn_accel.losses import LossEvaluator, LossPack
from pinn_accel.models import MLP
from pinn_accel.optim import make_lbfgs_optimizer
from pinn_accel.settings import configure_torch, resolve_device, set_seed
from pinn_accel.training import RelativeL2Metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experimental L-BFGS run that lets a controller/agent update loss "
            "weights between outer L-BFGS steps and reinitializes L-BFGS every step."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "burgers_tiny_loss_weight.json",
    )
    parser.add_argument("--data-path", type=Path, help="Override equation_params.data_path.")
    parser.add_argument("--sample-id", type=int, help="Override equation_params.sample_id.")
    parser.add_argument(
        "--target-time",
        type=float,
        help="Override equation_params.target_time.",
    )
    parser.add_argument("--controller", default="tiny_loss_weight")
    parser.add_argument("--reward", help="Override agent reward.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--lbfgs-lr", type=float)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts" / "lbfgs_agent_reinit")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--relative-l2-every", type=int, default=10)
    parser.add_argument("--agent-update-interval", type=int, default=1)
    parser.add_argument("--agent-warmup-steps", type=int, default=0)
    parser.add_argument("--controller-label", help="Output label.")
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Do not auto-run fixed baseline for baseline rewards.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _controller_params(cfg: ExperimentConfig, name: str, reward: str | None) -> dict[str, Any]:
    params = dict(cfg.controller_params.get(name, {}))
    if reward is not None:
        params["reward"] = reward
    return params


def _apply_equation_overrides(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
) -> ExperimentConfig:
    data = cfg.to_dict()
    equation_params = dict(data.get("equation_params", {}))
    if args.data_path is not None:
        equation_params["data_path"] = str(args.data_path.expanduser())
    if args.sample_id is not None:
        equation_params["sample_id"] = args.sample_id
    if args.target_time is not None:
        equation_params["target_time"] = args.target_time
    data["equation_params"] = equation_params
    return ExperimentConfig.from_dict(data)


def _training_cfg(cfg: ExperimentConfig, args: argparse.Namespace) -> TrainingConfig:
    kwargs: dict[str, Any] = {
        "steps": args.steps,
        "optimizer_mode": "lbfgs",
        "adam_steps": None,
        "lbfgs_steps": args.steps,
        "lbfgs_max_iter": args.lbfgs_max_iter,
        "log_every": args.log_every,
        "relative_l2_every": args.relative_l2_every,
        "agent_update_interval": args.agent_update_interval,
        "agent_warmup_steps": args.agent_warmup_steps,
        "compile_model": False,
    }
    if args.lbfgs_lr is not None:
        kwargs["lbfgs_lr"] = args.lbfgs_lr
    return replace(cfg.training, **kwargs)


def _build_relative_l2_metric(
    spec,
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
        raise ValueError(f"Reference solution shape must be {expected_shape}, got {u.shape}")
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
        "optimizer_reinitialized": [],
        "agent_reward": [],
        "agent_sigma": [],
        "agent_frozen": [],
        "lbfgs_closure_calls": [],
        "lbfgs_objective_before": [],
        "lbfgs_objective_after": [],
        "lbfgs_closure_first_loss": [],
        "lbfgs_closure_last_loss": [],
        "model_param_delta_norm": [],
        "elapsed_sec": [],
    }


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


def _weighted_objective(loss_pack: LossPack, weights_t: torch.Tensor) -> torch.Tensor:
    return torch.sum(weights_t.to(loss_pack.values.device) * loss_pack.values)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _clone_params(params: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [param.detach().clone() for param in params]


def _param_delta_norm(
    params: list[torch.nn.Parameter],
    before: list[torch.Tensor],
) -> float:
    total = torch.zeros((), dtype=torch.float32, device=params[0].device)
    with torch.no_grad():
        for param, old_value in zip(params, before):
            delta = param.detach().float() - old_value.to(param.device).float()
            total = total + torch.sum(delta * delta)
    return float(torch.sqrt(total).cpu().item())


def train_lbfgs_agent_reinit(
    *,
    spec,
    model_cfg,
    train_cfg: TrainingConfig,
    controller: WeightController,
    device: torch.device,
    seed: int,
    baseline_history: dict[str, Any] | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], WeightController, float]:
    if controller.trainable:
        raise ValueError(
            "This experiment supports agent/stateful weight controllers, not trainable "
            "controller parameters such as GradNorm."
        )

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
    controller.bind(component_names, np.ones(len(component_names), dtype=np.float32), device)
    model_params = list(model.parameters())
    relative_l2_metric = (
        _build_relative_l2_metric(spec, device, train_cfg.relative_l2_chunk_size)
        if train_cfg.relative_l2_every > 0
        else None
    )

    history = _empty_history(component_names, controller.name)
    history["batch_info"] = loss_evaluator.batch_info
    start_time = time.time()
    total_steps = int(train_cfg.steps)

    for step in range(1, total_steps + 1):
        step_start = time.time()
        progress = step / float(max(total_steps, 1))
        batches = loss_evaluator.draw_batches()

        loss_pack = loss_evaluator.compute(model, batches)
        objective, weights_t = controller.objective(
            loss_pack.values,
            model,
            step,
            update_state=True,
        )
        del objective
        fixed_weights = weights_t.detach().clone()
        lbfgs_objective_before = _scalar(_weighted_objective(loss_pack, fixed_weights))
        del loss_pack

        optimizer = make_lbfgs_optimizer(
            model_params,
            lr=train_cfg.lbfgs_lr,
            max_iter=train_cfg.lbfgs_max_iter,
            max_eval=train_cfg.lbfgs_max_eval,
            history_size=train_cfg.lbfgs_history_size,
            tolerance_grad=train_cfg.lbfgs_tolerance_grad,
            tolerance_change=train_cfg.lbfgs_tolerance_change,
            line_search_fn=train_cfg.lbfgs_line_search_fn,
        )
        model_params_before = _clone_params(model_params)
        lbfgs_closure_calls = 0
        lbfgs_closure_first_loss = None
        lbfgs_closure_last_loss = None

        def closure() -> torch.Tensor:
            nonlocal lbfgs_closure_calls
            nonlocal lbfgs_closure_first_loss, lbfgs_closure_last_loss
            lbfgs_closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            closure_pack = loss_evaluator.compute(model, batches)
            closure_objective = _weighted_objective(closure_pack, fixed_weights)
            closure_loss = _scalar(closure_objective)
            if lbfgs_closure_first_loss is None:
                lbfgs_closure_first_loss = closure_loss
            lbfgs_closure_last_loss = closure_loss
            closure_objective.backward()
            return closure_objective

        optimizer.step(closure)

        loss_pack = loss_evaluator.compute(model, batches)
        lbfgs_objective_after = _scalar(_weighted_objective(loss_pack, fixed_weights))
        raw_losses, weights_np, equal_total, weighted_total = _history_values(
            loss_pack,
            component_names,
            fixed_weights,
        )
        del loss_pack

        model_param_delta_norm = _param_delta_norm(model_params, model_params_before)
        relative_l2 = None
        if relative_l2_metric is not None and (
            step % train_cfg.relative_l2_every == 0 or step == total_steps
        ):
            relative_l2 = relative_l2_metric(model)

        snapshot = StepSnapshot(
            step=step,
            total=equal_total,
            losses=raw_losses,
            weights=weights_np,
            relative_l2=relative_l2,
            progress=progress,
            agent_progress=progress,
            done=step == total_steps,
        )
        extras = controller.after_step(snapshot, baseline_history)

        history["equal_weight_total"].append(equal_total)
        history["weighted_total"].append(weighted_total)
        history["relative_l2"].append(relative_l2)
        for name, value in zip(component_names, raw_losses):
            history["components"][name].append(float(value))
        history["weights"].append(weights_np.tolist())
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["progress"].append(progress)
        history["agent_progress"].append(progress)
        history["optimizer_phase"].append("lbfgs_agent_reinit")
        history["optimizer_reinitialized"].append(True)
        history["agent_reward"].append(extras.get("agent_reward"))
        history["agent_sigma"].append(extras.get("agent_sigma"))
        history["agent_frozen"].append(False)
        history["lbfgs_closure_calls"].append(lbfgs_closure_calls)
        history["lbfgs_objective_before"].append(lbfgs_objective_before)
        history["lbfgs_objective_after"].append(lbfgs_objective_after)
        history["lbfgs_closure_first_loss"].append(lbfgs_closure_first_loss)
        history["lbfgs_closure_last_loss"].append(lbfgs_closure_last_loss)
        history["model_param_delta_norm"].append(model_param_delta_norm)
        history["elapsed_sec"].append(time.time() - step_start)

        if train_cfg.log_every > 0 and (
            step == 1 or step % train_cfg.log_every == 0 or step == total_steps
        ):
            pieces = " ".join(
                f"{name}={history['components'][name][-1]:.3e}"
                for name in component_names
            )
            print(
                f"[{spec.name}/{controller.name}] phase=lbfgs_agent_reinit "
                f"step={step}/{total_steps} equal={equal_total:.3e} "
                f"weighted={weighted_total:.3e} "
                f"rel_l2={relative_l2 if relative_l2 is not None else float('nan'):.3e} "
                f"param_delta={model_param_delta_norm:.3e} "
                f"closure_calls={lbfgs_closure_calls} "
                f"lbfgs_before={lbfgs_objective_before:.3e} "
                f"lbfgs_after={lbfgs_objective_after:.3e} "
                f"reward={extras.get('agent_reward')} "
                f"{pieces}"
            )

    return model, history, controller, time.time() - start_time


def _series(values: list[Any]) -> np.ndarray:
    return np.array([np.nan if value is None else float(value) for value in values])


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_plots(history: dict[str, Any], plot_dir: Path) -> None:
    steps = np.arange(1, len(history["weighted_total"]) + 1)

    fig = plt.figure(figsize=(7, 4))
    plt.semilogy(steps, history["equal_weight_total"], label="equal")
    plt.semilogy(steps, history["weighted_total"], label="weighted")
    plt.xlabel("outer L-BFGS step")
    plt.ylabel("loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save_plot(fig, plot_dir / "loss_total.png")

    fig = plt.figure(figsize=(7, 4))
    for name in history["component_names"]:
        plt.semilogy(steps, history["components"][name], label=name)
    plt.xlabel("outer L-BFGS step")
    plt.ylabel("component loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save_plot(fig, plot_dir / "loss_components.png")

    weights = np.asarray(history["weights"], dtype=np.float64)
    fig = plt.figure(figsize=(7, 4))
    for idx, name in enumerate(history["component_names"]):
        plt.plot(steps, weights[:, idx], label=f"w_{name}")
    plt.xlabel("outer L-BFGS step")
    plt.ylabel("loss weight")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    _save_plot(fig, plot_dir / "weights.png")

    relative_l2 = _series(history.get("relative_l2", []))
    mask = np.isfinite(relative_l2)
    if np.any(mask):
        fig = plt.figure(figsize=(7, 4))
        plt.semilogy(steps[mask], relative_l2[mask])
        plt.xlabel("outer L-BFGS step")
        plt.ylabel("relative L2")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        _save_plot(fig, plot_dir / "relative_l2.png")

    fig = plt.figure(figsize=(7, 4))
    plt.semilogy(steps, history["model_param_delta_norm"])
    plt.xlabel("outer L-BFGS step")
    plt.ylabel("model parameter delta")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    _save_plot(fig, plot_dir / "param_delta.png")

    fig = plt.figure(figsize=(7, 4))
    plt.plot(steps, history["lbfgs_closure_calls"])
    plt.xlabel("outer L-BFGS step")
    plt.ylabel("closure calls")
    plt.grid(True, ls="--", alpha=0.3)
    _save_plot(fig, plot_dir / "closure_calls.png")

    rewards = _series(history.get("agent_reward", []))
    mask = np.isfinite(rewards)
    if np.any(mask):
        fig = plt.figure(figsize=(7, 4))
        plt.plot(steps[mask], rewards[mask], marker="o", markersize=3)
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.35)
        plt.xlabel("outer L-BFGS step")
        plt.ylabel("agent reward")
        plt.grid(True, ls="--", alpha=0.3)
        _save_plot(fig, plot_dir / "agent_reward.png")

    sigmas = _series(history.get("agent_sigma", []))
    mask = np.isfinite(sigmas)
    if np.any(mask):
        fig = plt.figure(figsize=(7, 4))
        plt.plot(steps[mask], sigmas[mask])
        plt.xlabel("outer L-BFGS step")
        plt.ylabel("agent sigma")
        plt.grid(True, ls="--", alpha=0.3)
        _save_plot(fig, plot_dir / "agent_sigma.png")


def save_comparison_plot(
    histories: dict[str, dict[str, Any]],
    plot_dir: Path,
) -> None:
    if len(histories) < 2:
        return

    def plot_metric(
        key: str,
        ylabel: str,
        filename: str,
        *,
        log_y: bool,
    ) -> None:
        fig = plt.figure(figsize=(7, 4))
        has_values = False
        for label, history in histories.items():
            values = _series(history.get(key, []))
            if values.size == 0:
                continue
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            steps = np.arange(1, len(values) + 1)
            plot = plt.semilogy if log_y else plt.plot
            plot(steps[mask], values[mask], label=label)
            has_values = True
        if not has_values:
            plt.close(fig)
            return
        plt.xlabel("outer L-BFGS step")
        plt.ylabel(ylabel)
        plt.grid(True, which="both" if log_y else "major", ls="--", alpha=0.3)
        plt.legend()
        _save_plot(fig, plot_dir / filename)

    plot_metric(
        "equal_weight_total",
        "equal-weight loss",
        "comparison_equal_loss.png",
        log_y=True,
    )
    plot_metric(
        "weighted_total",
        "weighted loss",
        "comparison_weighted_loss.png",
        log_y=True,
    )
    plot_metric(
        "relative_l2",
        "relative L2",
        "comparison_relative_l2.png",
        log_y=True,
    )
    plot_metric(
        "model_param_delta_norm",
        "model parameter delta",
        "comparison_param_delta.png",
        log_y=True,
    )
    plot_metric(
        "lbfgs_closure_calls",
        "closure calls",
        "comparison_closure_calls.png",
        log_y=False,
    )

    component_names = list(next(iter(histories.values()))["component_names"])
    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4.2 * len(component_names), 3.8),
    )
    axes = np.atleast_1d(axes)
    for axis, component in zip(axes, component_names):
        for label, history in histories.items():
            values = np.asarray(history["components"][component], dtype=np.float64)
            steps = np.arange(1, len(values) + 1)
            axis.semilogy(steps, values, label=label)
        axis.set_title(component)
        axis.set_xlabel("outer step")
        axis.grid(True, which="both", ls="--", alpha=0.3)
    axes[0].set_ylabel("component loss")
    axes[0].legend()
    _save_plot(fig, plot_dir / "comparison_components.png")

    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4.2 * len(component_names), 3.8),
    )
    axes = np.atleast_1d(axes)
    for idx, (axis, component) in enumerate(zip(axes, component_names)):
        for label, history in histories.items():
            weights = np.asarray(history["weights"], dtype=np.float64)
            steps = np.arange(1, weights.shape[0] + 1)
            axis.plot(steps, weights[:, idx], label=label)
        axis.set_title(f"w_{component}")
        axis.set_xlabel("outer step")
        axis.grid(True, ls="--", alpha=0.3)
    axes[0].set_ylabel("loss weight")
    axes[0].legend()
    _save_plot(fig, plot_dir / "comparison_weights.png")

    fig = plt.figure(figsize=(7, 4))
    has_rewards = False
    for label, history in histories.items():
        rewards = _series(history.get("agent_reward", []))
        mask = np.isfinite(rewards)
        if not np.any(mask):
            continue
        steps = np.arange(1, len(rewards) + 1)
        plt.plot(steps[mask], rewards[mask], label=label)
        has_rewards = True
    if has_rewards:
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.35)
        plt.xlabel("outer L-BFGS step")
        plt.ylabel("agent reward")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        _save_plot(fig, plot_dir / "comparison_agent_reward.png")
    else:
        plt.close(fig)

    fig = plt.figure(figsize=(7, 4))
    has_sigmas = False
    for label, history in histories.items():
        sigmas = _series(history.get("agent_sigma", []))
        mask = np.isfinite(sigmas)
        if not np.any(mask):
            continue
        steps = np.arange(1, len(sigmas) + 1)
        plt.plot(steps[mask], sigmas[mask], label=label)
        has_sigmas = True
    if has_sigmas:
        plt.xlabel("outer L-BFGS step")
        plt.ylabel("agent sigma")
        plt.grid(True, ls="--", alpha=0.3)
        plt.legend()
        _save_plot(fig, plot_dir / "comparison_agent_sigma.png")
    else:
        plt.close(fig)


def _last_finite(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return finite[-1] if finite else None


def main() -> None:
    args = parse_args()
    cfg = _apply_equation_overrides(ExperimentConfig.from_file(args.config), args)
    params = _controller_params(cfg, args.controller, args.reward)
    training = _training_cfg(cfg, args)
    reward_name = str(params.get("reward", ""))
    if "relative_l2" in reward_name and training.relative_l2_every != 1:
        print("relative_l2 reward selected; forcing --relative-l2-every 1")
        training = replace(training, relative_l2_every=1)

    configure_torch()
    device = resolve_device(args.device or cfg.device)
    seed = cfg.seed if args.seed is None else args.seed
    spec = get_equation(cfg.equation, **cfg.equation_params)
    store = ArtifactStore.create(args.out)
    store.save_json(store.root / "config.json", cfg.to_dict())
    store.save_json(
        store.root / "lbfgs_agent_reinit_config.json",
        {
            "controller": args.controller,
            "controller_params": params,
            "training": asdict(training),
            "device": str(device),
            "seed": seed,
            "note": "L-BFGS optimizer is reinitialized every outer step.",
        },
    )
    print(f"Device: {device}")
    print(f"Run: {store.root}")

    histories: dict[str, dict[str, Any]] = {}
    baseline_history = None
    if (
        controller_needs_baseline(args.controller, params)
        and not args.no_baseline
    ):
        print("Baseline reward detected; running fixed L-BFGS baseline first.")
        fixed_controller = make_controller(
            "fixed",
            {},
            update_interval=training.agent_update_interval,
            warmup_steps=training.agent_warmup_steps,
        )
        _, baseline_history, _, baseline_elapsed = train_lbfgs_agent_reinit(
            spec=spec,
            model_cfg=cfg.model,
            train_cfg=training,
            controller=fixed_controller,
            device=device,
            seed=seed,
        )
        histories["fixed_baseline"] = baseline_history
        method_dir = store.method_dir(spec.name, "fixed_baseline")
        store.save_json(method_dir / "history.json", baseline_history)
        if not args.no_plots:
            save_plots(baseline_history, method_dir / "plots")
        print(f"[{spec.name}/fixed_baseline] elapsed={baseline_elapsed:.1f}s")

    controller = make_controller(
        args.controller,
        params,
        update_interval=training.agent_update_interval,
        warmup_steps=training.agent_warmup_steps,
    )
    if controller_needs_baseline(args.controller, params) and baseline_history is None:
        raise ValueError(
            f"{args.controller} with reward={params.get('reward')} requires baseline. "
            "Remove --no-baseline or choose a self-reward."
        )

    model, history, controller, elapsed = train_lbfgs_agent_reinit(
        spec=spec,
        model_cfg=cfg.model,
        train_cfg=training,
        controller=controller,
        device=device,
        seed=seed,
        baseline_history=baseline_history,
    )
    del model, controller

    label = args.controller_label or args.controller
    histories[label] = history
    method_dir = store.method_dir(spec.name, label)
    store.save_json(method_dir / "history.json", history)
    if not args.no_plots:
        save_plots(history, method_dir / "plots")
        save_comparison_plot(histories, store.equation_dir(spec.name) / "comparison" / "plots")

    summary = {
        label: {
            "elapsed_sec": elapsed,
            "final_equal_weight_total": history["equal_weight_total"][-1],
            "final_weighted_total": history["weighted_total"][-1],
            "final_relative_l2": _last_finite(history.get("relative_l2", [])),
            "final_weights": history["weights"][-1],
            "final_agent_reward": _last_finite(history.get("agent_reward", [])),
            "final_agent_sigma": _last_finite(history.get("agent_sigma", [])),
            "total_closure_calls": int(sum(history["lbfgs_closure_calls"])),
        }
    }
    if baseline_history is not None:
        summary["fixed_baseline"] = {
            "final_equal_weight_total": baseline_history["equal_weight_total"][-1],
            "final_weighted_total": baseline_history["weighted_total"][-1],
            "final_relative_l2": _last_finite(baseline_history.get("relative_l2", [])),
            "total_closure_calls": int(sum(baseline_history["lbfgs_closure_calls"])),
        }
    store.save_json(store.equation_dir(spec.name) / "summary.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2))
    print(f"History: {method_dir / 'history.json'}")
    if not args.no_plots:
        print(f"Plots: {method_dir / 'plots'}")


if __name__ == "__main__":
    main()
