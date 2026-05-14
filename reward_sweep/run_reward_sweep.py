from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pinn_accel.artifacts import ArtifactStore, to_jsonable  # noqa: E402
from pinn_accel.checkpoints import build_result_checkpoint  # noqa: E402
from pinn_accel.config import ExperimentConfig  # noqa: E402
from pinn_accel.controllers import make_controller  # noqa: E402
from pinn_accel.equations import get_equation  # noqa: E402
from pinn_accel.equations.base import EquationSpec  # noqa: E402
from pinn_accel.plots import save_history_plots, save_solution_plot  # noqa: E402
from pinn_accel.settings import configure_torch, resolve_device  # noqa: E402
from pinn_accel.training import TrainResult, train_one  # noqa: E402


ALL_REWARDS = [
    "log_ratio",
    "relative_improvement",
    "component_relative_improvement",
    "baseline_gap",
    "normalized_baseline_gap",
    "normalized_baseline_gap_delta",
    "relative_l2_improvement",
    "relative_l2_baseline_gap",
    "relative_l2_baseline_gap_delta",
    "log_normalized_baseline_gap",
    "component_baseline_gap",
    "worst_component_baseline_gap",
    "loss_l2_hybrid",
    "progressive_loss_l2_hybrid",
    "worst_component_relative_improvement",
    "component_balance_penalty",
    "relative_l2_log_improvement",
    "loss_l2_self_hybrid",
    "running_best_l2_reward",
]


REWARD_COLORS = {
    reward: color
    for reward, color in zip(
        ALL_REWARDS,
        [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#17becf",
            "#bcbd22",
            "#aec7e8",
            "#ffbb78",
            "#98df8a",
            "#ff9896",
            "#c5b0d5",
            "#c49c94",
            "#f7b6d2",
            "#c7c7c7",
            "#dbdb8d",
        ],
    )
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed + agent reward sweep.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "reward_sweep" / "base_config.json",
        help="Base sweep config.",
    )
    parser.add_argument("--out", type=Path, help="Output directory.")
    parser.add_argument("--rewards", help="Comma-separated reward names.")
    parser.add_argument("--steps", type=int, help="Override training steps.")
    parser.add_argument(
        "--optimizer-mode",
        choices=["adam", "adam_lbfgs", "lbfgs"],
        help="PINN optimizer schedule.",
    )
    parser.add_argument("--adam-steps", type=int, help="Adam phase steps.")
    parser.add_argument("--lbfgs-steps", type=int, help="L-BFGS phase steps.")
    parser.add_argument("--lbfgs-max-iter", type=int, help="L-BFGS max_iter per step.")
    parser.add_argument(
        "--agent-update-interval",
        type=int,
        help="Agent update interval in active training steps.",
    )
    parser.add_argument(
        "--agent-warmup-steps",
        type=int,
        help="Number of active agent steps before agent updates start.",
    )
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=None)
    return parser.parse_args()


def _validate_rewards(rewards: list[str]) -> list[str]:
    unknown = sorted(set(rewards) - set(ALL_REWARDS))
    if unknown:
        known = ", ".join(ALL_REWARDS)
        raise ValueError(f"Unknown rewards: {unknown}. Known rewards: {known}")
    return rewards


def _selected_rewards(config_payload: dict[str, Any], value: str | None) -> list[str]:
    if value is not None:
        rewards = [item.strip() for item in value.split(",") if item.strip()]
        return _validate_rewards(rewards)
    rewards = config_payload.get("sweep_rewards", ALL_REWARDS)
    if not isinstance(rewards, list) or not all(isinstance(item, str) for item in rewards):
        raise ValueError("sweep_rewards must be a list of reward names")
    return _validate_rewards(rewards)


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    data = cfg.to_dict()
    data["controllers"] = ["fixed", "tiny_loss_weight"]
    if args.out is not None:
        data["output_dir"] = str(args.out)
    if args.save_plots is not None:
        data["save_plots"] = args.save_plots
    if args.steps is not None:
        data["training"]["steps"] = args.steps
    if args.optimizer_mode is not None:
        data["training"]["optimizer_mode"] = args.optimizer_mode
    if args.adam_steps is not None:
        data["training"]["adam_steps"] = args.adam_steps
    if args.lbfgs_steps is not None:
        data["training"]["lbfgs_steps"] = args.lbfgs_steps
    if args.lbfgs_max_iter is not None:
        data["training"]["lbfgs_max_iter"] = args.lbfgs_max_iter
    if args.agent_update_interval is not None:
        data["training"]["agent_update_interval"] = args.agent_update_interval
    if args.agent_warmup_steps is not None:
        data["training"]["agent_warmup_steps"] = args.agent_warmup_steps
    if args.device is not None:
        data["device"] = args.device
    if args.seed is not None:
        data["seed"] = args.seed
    return ExperimentConfig.from_dict(data)


def _controller_params(cfg: ExperimentConfig, name: str) -> dict[str, Any]:
    return dict(cfg.controller_params.get(name, {}))


def _save_result(
    *,
    store: ArtifactStore,
    cfg: ExperimentConfig,
    result: TrainResult,
    spec: EquationSpec,
    label: str,
    device: torch.device,
) -> None:
    method_dir = store.method_dir(spec.name, label)
    store.save_history(spec.name, label, result.history)
    if result.history.get("batch_info") is not None:
        store.save_json(method_dir / "batch_info.json", result.history["batch_info"])
    checkpoint = build_result_checkpoint(
        equation_name=spec.name,
        label=label,
        result=result,
        model_config=cfg.model,
        training_config=cfg.training,
    )
    store.save_checkpoint(spec.name, label, checkpoint)
    if checkpoint.get("agent") is not None:
        store.save_agent_checkpoint(spec.name, label, checkpoint["agent"])
    if cfg.save_plots:
        save_history_plots(result.history, method_dir / "plots")
        save_solution_plot(
            result.model,
            spec,
            device,
            method_dir / "plots",
            n=cfg.plot_grid,
        )


def _finite_series(values: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    series = np.array(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )
    steps = np.arange(1, len(series) + 1)
    return steps[np.isfinite(series)], series[np.isfinite(series)]


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path = path.with_suffix(".pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_metric(
    fixed_history: dict[str, Any],
    reward_histories: dict[str, dict[str, Any]],
    key: str,
    ylabel: str,
    path: Path,
    *,
    log_y: bool,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    fixed_steps, fixed_values = _finite_series(fixed_history.get(key, []))
    if fixed_values.size:
        plot = plt.semilogy if log_y else plt.plot
        plot(fixed_steps, fixed_values, label="fixed", color="black", linewidth=2.0)
    for reward, history in reward_histories.items():
        steps, values = _finite_series(history.get(key, []))
        if not values.size:
            continue
        plot = plt.semilogy if log_y else plt.plot
        plot(steps, values, label=reward, color=REWARD_COLORS.get(reward))
    plt.xlabel("step")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=8)
    _save_figure(fig, path)


def _plot_agent_rewards(
    reward_histories: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    for reward, history in reward_histories.items():
        steps, values = _finite_series(history.get("agent_reward", []))
        if not values.size:
            continue
        plt.plot(steps, values, label=reward, color=REWARD_COLORS.get(reward))
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    plt.xlabel("step")
    plt.ylabel("agent reward")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(fontsize=8)
    _save_figure(fig, path)


def _plot_agent_sigmas(
    reward_histories: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(8, 5))
    has_values = False
    for reward, history in reward_histories.items():
        steps, values = _finite_series(history.get("agent_sigma", []))
        if not values.size:
            continue
        has_values = True
        plt.plot(steps, values, label=reward, color=REWARD_COLORS.get(reward))
    if not has_values:
        plt.close(fig)
        return
    plt.xlabel("step")
    plt.ylabel("sigma")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend(fontsize=8)
    _save_figure(fig, path)


def _plot_weights(
    reward_histories: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    if not reward_histories:
        return
    names = list(next(iter(reward_histories.values()))["component_names"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.3 * len(names), 3.8))
    axes = np.atleast_1d(axes)
    for idx, (axis, name) in enumerate(zip(axes, names)):
        for reward, history in reward_histories.items():
            weights = np.asarray(history["weights"], dtype=np.float64)
            axis.plot(weights[:, idx], label=reward, color=REWARD_COLORS.get(reward))
        axis.set_title(f"w_{name}")
        axis.set_xlabel("step")
        axis.grid(True, ls="--", alpha=0.3)
    axes[0].set_ylabel("weight")
    axes[0].legend(fontsize=7)
    _save_figure(fig, path)


def _reference_grid(spec: EquationSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if spec.reference_solver is None:
        return None
    x, t, u = spec.solve_reference()
    if u.shape == (len(t), len(x)):
        u = u.T
    if u.shape != (len(x), len(t)):
        raise ValueError(f"Reference solution shape must be {(len(x), len(t))}, got {u.shape}")
    return x, t, u


def _predict_slice(
    model: torch.nn.Module,
    x: np.ndarray,
    t_value: float,
    device: torch.device,
    chunk_size: int = 65536,
) -> np.ndarray:
    xt_np = np.stack([x, np.full_like(x, t_value)], axis=1)
    xt = torch.tensor(xt_np, dtype=torch.float32, device=device)
    values: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, xt.shape[0], chunk_size):
            stop = min(start + chunk_size, xt.shape[0])
            values.append(model(xt[start:stop]).reshape(-1).detach().cpu().numpy())
    return np.concatenate(values)


def _plot_solution_slices(
    fixed_model: torch.nn.Module,
    reward_results: dict[str, TrainResult],
    spec: EquationSpec,
    device: torch.device,
    times: list[float],
    path: Path,
) -> None:
    if not times:
        return
    reference = _reference_grid(spec)
    if reference is None:
        x = np.linspace(spec.x_min, spec.x_max, 512, dtype=np.float64)
        t_ref = None
        u_ref = None
    else:
        x, t_ref, u_ref = reference
    valid_times = [
        float(value)
        for value in times
        if spec.t_min - 1e-12 <= float(value) <= spec.t_max + 1e-12
    ]
    if not valid_times:
        return
    ncols = min(2, len(valid_times))
    nrows = int(np.ceil(len(valid_times) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.8 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.25)
    axes_flat = axes.reshape(-1)
    for plot_idx, (axis, requested_time) in enumerate(zip(axes_flat, valid_times)):
        plot_time = requested_time
        if t_ref is not None and u_ref is not None:
            time_index = int(np.argmin(np.abs(t_ref - requested_time)))
            plot_time = float(t_ref[time_index])
            axis.plot(
                x,
                u_ref[:, time_index],
                label="reference",
                color="black",
                linestyle="--",
                linewidth=2.0,
            )
        fixed_prediction = _predict_slice(fixed_model, x, plot_time, device)
        axis.plot(x, fixed_prediction, label="fixed", color="black", alpha=0.6)
        for reward, result in reward_results.items():
            prediction = _predict_slice(result.model, x, plot_time, device)
            axis.plot(
                x,
                prediction,
                label=reward,
                color=REWARD_COLORS.get(reward),
                alpha=0.9,
            )
        axis.set_title(f"t={plot_time:.3g}")
        if plot_idx // ncols == nrows - 1:
            axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.grid(True, ls="--", alpha=0.3)
    for axis in axes_flat[len(valid_times) :]:
        axis.axis("off")
    axes_flat[0].legend(fontsize=7)
    _save_figure(fig, path)


def _last_finite(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return finite[-1] if finite else None


def _save_sweep_plots(
    fixed_result: TrainResult,
    reward_results: dict[str, TrainResult],
    spec: EquationSpec,
    device: torch.device,
    cfg: ExperimentConfig,
    plot_dir: Path,
) -> None:
    reward_histories = {
        reward: result.history
        for reward, result in reward_results.items()
    }
    _plot_metric(
        fixed_result.history,
        reward_histories,
        "equal_weight_total",
        "equal-weight loss",
        plot_dir / "reward_sweep_equal_loss.pdf",
        log_y=True,
    )
    _plot_metric(
        fixed_result.history,
        reward_histories,
        "relative_l2",
        "relative L2 error",
        plot_dir / "reward_sweep_relative_l2.pdf",
        log_y=True,
    )
    _plot_agent_rewards(reward_histories, plot_dir / "reward_sweep_agent_reward.pdf")
    _plot_agent_sigmas(reward_histories, plot_dir / "reward_sweep_agent_sigma.pdf")
    _plot_weights(reward_histories, plot_dir / "reward_sweep_weights.pdf")
    _plot_solution_slices(
        fixed_result.model,
        reward_results,
        spec,
        device,
        cfg.solution_slice_times,
        plot_dir / "reward_sweep_solution_slices.pdf",
    )


def main() -> None:
    args = parse_args()
    config_payload = json.loads(args.config.read_text(encoding="utf-8"))
    rewards = _selected_rewards(config_payload, args.rewards)
    config_payload.pop("sweep_rewards", None)
    cfg = _apply_overrides(ExperimentConfig.from_dict(config_payload), args)

    configure_torch()
    device = resolve_device(cfg.device)
    spec = get_equation(cfg.equation, **cfg.equation_params)
    store = ArtifactStore.create(cfg.output_dir)
    store.save_json(store.root / "reward_sweep_config.json", cfg.to_dict())
    store.save_json(store.root / "reward_sweep_rewards.json", rewards)
    print(f"Device: {device}")
    print(f"Run: {store.root}")

    fixed_controller = make_controller(
        "fixed",
        _controller_params(cfg, "fixed"),
        update_interval=cfg.training.agent_update_interval,
        warmup_steps=cfg.training.agent_warmup_steps,
    )
    fixed_result = train_one(
        spec=spec,
        model_cfg=cfg.model,
        train_cfg=cfg.training,
        controller=fixed_controller,
        device=device,
        seed=cfg.seed,
    )
    _save_result(
        store=store,
        cfg=cfg,
        result=fixed_result,
        spec=spec,
        label="fixed",
        device=device,
    )

    reward_results: dict[str, TrainResult] = {}
    for reward in rewards:
        print(f"[{spec.name}/agent] reward={reward}")
        params = _controller_params(cfg, "tiny_loss_weight")
        params["reward"] = reward
        controller = make_controller(
            "tiny_loss_weight",
            params,
            update_interval=cfg.training.agent_update_interval,
            warmup_steps=cfg.training.agent_warmup_steps,
        )
        result = train_one(
            spec=spec,
            model_cfg=cfg.model,
            train_cfg=cfg.training,
            controller=controller,
            device=device,
            seed=cfg.seed,
            baseline_history=fixed_result.history,
        )
        reward_results[reward] = result
        _save_result(
            store=store,
            cfg=cfg,
            result=result,
            spec=spec,
            label=f"agent_{reward}",
            device=device,
        )

    plot_dir = store.equation_dir(spec.name) / "reward_sweep" / "plots"
    if cfg.save_plots:
        _save_sweep_plots(fixed_result, reward_results, spec, device, cfg, plot_dir)

    summary = {
        "fixed": {
            "final_equal_weight_total": fixed_result.history["equal_weight_total"][-1],
            "final_relative_l2": _last_finite(fixed_result.history.get("relative_l2", [])),
        },
        "rewards": {
            reward: {
                "final_equal_weight_total": result.history["equal_weight_total"][-1],
                "final_relative_l2": _last_finite(result.history.get("relative_l2", [])),
                "final_weights": result.history["weights"][-1],
                "final_agent_reward": _last_finite(result.history.get("agent_reward", [])),
                "final_agent_sigma": _last_finite(result.history.get("agent_sigma", [])),
            }
            for reward, result in reward_results.items()
        },
    }
    summary_path = store.equation_dir(spec.name) / "reward_sweep" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(to_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    print(f"Summary: {summary_path}")
    print(f"Plots: {plot_dir}")


if __name__ == "__main__":
    main()
