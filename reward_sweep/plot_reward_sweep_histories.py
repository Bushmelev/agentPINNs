from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ALL_REWARDS = [
    "fixed",
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


DEFAULT_COLORS = {
    "fixed": "#111111",
    "log_ratio": "#1f77b4",
    "relative_improvement": "#ff7f0e",
    "component_relative_improvement": "#2ca02c",
    "baseline_gap": "#d62728",
    "normalized_baseline_gap": "#9467bd",
    "normalized_baseline_gap_delta": "#8c564b",
    "relative_l2_improvement": "#e377c2",
    "relative_l2_baseline_gap": "#7f7f7f",
    "relative_l2_baseline_gap_delta": "#17becf",
    "log_normalized_baseline_gap": "#bcbd22",
    "component_baseline_gap": "#aec7e8",
    "worst_component_baseline_gap": "#ffbb78",
    "loss_l2_hybrid": "#98df8a",
    "progressive_loss_l2_hybrid": "#ff9896",
    "worst_component_relative_improvement": "#c5b0d5",
    "component_balance_penalty": "#c49c94",
    "relative_l2_log_improvement": "#f7b6d2",
    "loss_l2_self_hybrid": "#c7c7c7",
    "running_best_l2_reward": "#dbdb8d",
}


EXTRA_COLORS = [
    "#393b79",
    "#637939",
    "#8c6d31",
    "#843c39",
    "#7b4173",
    "#3182bd",
    "#31a354",
    "#756bb1",
    "#636363",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot reward sweep comparison curves from saved history.json files."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Sweep root or equation folder, e.g. .../<timestamp> or .../<timestamp>/burgers.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output plot directory. Default: <equation>/reward_sweep/history_plots.",
    )
    parser.add_argument(
        "--color-map",
        type=Path,
        help=(
            "JSON file with reward -> color mapping. If omitted, a reward_colors.json "
            "file is written in the output directory."
        ),
    )
    parser.add_argument(
        "--formats",
        default="pdf",
        help="Comma-separated output formats, e.g. pdf,png.",
    )
    parser.add_argument(
        "--rewards",
        help="Optional comma-separated subset of rewards to plot. 'fixed' may be included.",
    )
    parser.add_argument(
        "--no-fixed",
        action="store_true",
        help="Do not include fixed baseline on comparison plots.",
    )
    parser.add_argument(
        "--reward-window",
        type=int,
        default=25,
        help="Moving-average window for agent reward plots.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def resolve_equation_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "history.json").is_file():
        return root.parent
    if root.name == "reward_sweep":
        return root.parent
    if (root / "fixed" / "history.json").is_file() or list(root.glob("agent_*/history.json")):
        return root
    candidates = [
        child
        for child in root.iterdir()
        if child.is_dir()
        and ((child / "fixed" / "history.json").is_file() or list(child.glob("agent_*/history.json")))
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No reward sweep histories found under {root}")
    names = ", ".join(str(path) for path in candidates)
    raise ValueError(f"Multiple equation folders found. Pass one explicitly: {names}")


def reward_name(run_dir: Path, history: dict[str, Any]) -> str:
    name = run_dir.name
    if name == "fixed":
        return "fixed"
    if name.startswith("agent_"):
        return name[len("agent_") :]
    controller = history.get("controller")
    if isinstance(controller, str) and "_" in controller:
        return controller.split("_", 1)[1]
    return name


def load_histories(equation_dir: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for history_path in sorted(equation_dir.glob("*/history.json")):
        run_dir = history_path.parent
        if run_dir.name == "reward_sweep":
            continue
        history = load_json(history_path)
        histories[reward_name(run_dir, history)] = history
    if not histories:
        raise FileNotFoundError(f"No history.json files found under {equation_dir}")
    return histories


def selected_rewards(histories: dict[str, dict[str, Any]], requested: str | None, no_fixed: bool) -> list[str]:
    available = set(histories)
    if requested:
        selected = [item.strip() for item in requested.split(",") if item.strip()]
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"Requested rewards are missing: {missing}")
    else:
        selected = [reward for reward in ALL_REWARDS if reward in available]
        selected.extend(sorted(available - set(selected)))
    if no_fixed:
        selected = [reward for reward in selected if reward != "fixed"]
    return selected


def load_color_map(path: Path | None, rewards: list[str]) -> dict[str, str]:
    color_map = dict(DEFAULT_COLORS)
    if path is not None and path.is_file():
        loaded = load_json(path)
        color_map.update({str(key): str(value) for key, value in loaded.items()})
    extra_idx = 0
    for reward in rewards:
        if reward in color_map:
            continue
        color_map[reward] = EXTRA_COLORS[extra_idx % len(EXTRA_COLORS)]
        extra_idx += 1
    return {reward: color_map[reward] for reward in rewards}


def save_color_map(path: Path, color_map: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(color_map, indent=2), encoding="utf-8")


def series(values: list[Any]) -> np.ndarray:
    return np.asarray(
        [np.nan if value is None else float(value) for value in values],
        dtype=np.float64,
    )


def phase_start(history: dict[str, Any], phase: str) -> int | None:
    for idx, value in enumerate(history.get("optimizer_phase", [])):
        if value == phase:
            return idx
    return None


def phase_slice(history: dict[str, Any], phase: str | None) -> slice:
    if phase is None:
        return slice(None)
    if phase == "adam":
        stop = phase_start(history, "lbfgs")
        return slice(0, stop)
    if phase == "lbfgs":
        start = phase_start(history, "lbfgs")
        if start is None:
            return slice(0, 0)
        return slice(start, None)
    raise ValueError(f"Unknown phase: {phase}")


def phase_x(values: np.ndarray, phase: str | None, history: dict[str, Any]) -> np.ndarray:
    if phase == "lbfgs":
        return np.arange(1, len(values) + 1)
    start = 0 if phase != "lbfgs" else (phase_start(history, "lbfgs") or 0)
    return np.arange(start + 1, start + len(values) + 1)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    result = np.full_like(values, np.nan, dtype=np.float64)
    finite = np.isfinite(values)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        chunk = values[start : idx + 1]
        mask = finite[start : idx + 1]
        if np.any(mask):
            result[idx] = float(np.mean(chunk[mask]))
    return result


def legend_right(axis: plt.Axes, fontsize: int = 8) -> None:
    axis.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=fontsize,
        frameon=False,
    )


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric(
    histories: dict[str, dict[str, Any]],
    rewards: list[str],
    color_map: dict[str, str],
    out_dir: Path,
    formats: list[str],
    *,
    metric: str,
    ylabel: str,
    stem: str,
    phase: str | None,
    log_y: bool = True,
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for reward in rewards:
        history = histories[reward]
        raw = series(history.get(metric, []))
        values = raw[phase_slice(history, phase)]
        if not values.size:
            continue
        x = phase_x(values, phase, history)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        plot = axis.semilogy if log_y else axis.plot
        linewidth = 2.6 if reward == "fixed" else 1.8
        plot(x[mask], values[mask], label=reward, color=color_map[reward], linewidth=linewidth)
    axis.set_xlabel("L-BFGS outer step" if phase == "lbfgs" else "step")
    axis.set_ylabel(ylabel)
    axis.grid(True, which="both", ls="--", alpha=0.3)
    legend_right(axis)
    phase_label = "full" if phase is None else phase
    save(fig, out_dir, f"{phase_label}_{stem}", formats)


def plot_components(
    histories: dict[str, dict[str, Any]],
    rewards: list[str],
    color_map: dict[str, str],
    out_dir: Path,
    formats: list[str],
    *,
    phase: str | None,
) -> None:
    component_names = list(next(iter(histories.values()))["component_names"])
    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4.8 * len(component_names) + 4.0, 4.2),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    handles = []
    labels = []
    for axis, component in zip(axes_flat, component_names):
        for reward in rewards:
            history = histories[reward]
            values = series(history["components"][component])[phase_slice(history, phase)]
            if not values.size:
                continue
            x = phase_x(values, phase, history)
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            (line,) = axis.semilogy(
                x[mask],
                values[mask],
                label=reward,
                color=color_map[reward],
                linewidth=2.6 if reward == "fixed" else 1.6,
            )
            if component == component_names[0]:
                handles.append(line)
                labels.append(reward)
        axis.set_title(component)
        axis.set_xlabel("L-BFGS outer step" if phase == "lbfgs" else "step")
        axis.grid(True, which="both", ls="--", alpha=0.3)
    axes_flat[0].set_ylabel("component loss")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    phase_label = "full" if phase is None else phase
    save(fig, out_dir, f"{phase_label}_component_losses", formats)


def plot_weights(
    histories: dict[str, dict[str, Any]],
    rewards: list[str],
    color_map: dict[str, str],
    out_dir: Path,
    formats: list[str],
    *,
    phase: str | None,
) -> None:
    component_names = list(next(iter(histories.values()))["component_names"])
    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4.8 * len(component_names) + 4.0, 4.2),
        squeeze=False,
    )
    axes_flat = axes.reshape(-1)
    handles = []
    labels = []
    for axis, component in zip(axes_flat, component_names):
        comp_idx = component_names.index(component)
        for reward in rewards:
            history = histories[reward]
            weights = np.asarray(history.get("weights", []), dtype=np.float64)
            if weights.size == 0:
                continue
            values = weights[:, comp_idx][phase_slice(history, phase)]
            if not values.size:
                continue
            x = phase_x(values, phase, history)
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            (line,) = axis.plot(
                x[mask],
                values[mask],
                label=reward,
                color=color_map[reward],
                linewidth=2.6 if reward == "fixed" else 1.6,
            )
            if component == component_names[0]:
                handles.append(line)
                labels.append(reward)
        axis.set_title(f"w_{component}")
        axis.set_xlabel("L-BFGS outer step" if phase == "lbfgs" else "step")
        axis.grid(True, ls="--", alpha=0.3)
    axes_flat[0].set_ylabel("loss weight")
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )
    phase_label = "full" if phase is None else phase
    save(fig, out_dir, f"{phase_label}_weights", formats)


def plot_agent_reward(
    histories: dict[str, dict[str, Any]],
    rewards: list[str],
    color_map: dict[str, str],
    out_dir: Path,
    formats: list[str],
    *,
    window: int,
) -> None:
    agent_rewards = [reward for reward in rewards if reward != "fixed"]
    for smoothed in (False, True):
        fig, axis = plt.subplots(figsize=(10, 5))
        has_values = False
        for reward in agent_rewards:
            history = histories[reward]
            values = series(history.get("agent_reward", []))[phase_slice(history, "adam")]
            if smoothed:
                values = moving_average(values, window)
            if not values.size:
                continue
            x = np.arange(1, len(values) + 1)
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            has_values = True
            axis.plot(
                x[mask],
                values[mask],
                label=reward,
                color=color_map[reward],
                linewidth=1.8,
            )
        if not has_values:
            plt.close(fig)
            continue
        axis.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
        axis.set_xlabel("Adam step")
        axis.set_ylabel(f"agent reward MA({window})" if smoothed else "agent reward")
        axis.grid(True, ls="--", alpha=0.3)
        legend_right(axis)
        stem = "adam_agent_reward_ma" if smoothed else "adam_agent_reward"
        save(fig, out_dir, stem, formats)


def plot_agent_sigma(
    histories: dict[str, dict[str, Any]],
    rewards: list[str],
    color_map: dict[str, str],
    out_dir: Path,
    formats: list[str],
) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    has_values = False
    for reward in [item for item in rewards if item != "fixed"]:
        history = histories[reward]
        values = series(history.get("agent_sigma", []))[phase_slice(history, "adam")]
        if not values.size:
            continue
        x = np.arange(1, len(values) + 1)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        has_values = True
        axis.plot(x[mask], values[mask], label=reward, color=color_map[reward], linewidth=1.8)
    if not has_values:
        plt.close(fig)
        return
    axis.set_xlabel("Adam step")
    axis.set_ylabel("agent sigma")
    axis.grid(True, ls="--", alpha=0.3)
    legend_right(axis)
    save(fig, out_dir, "adam_agent_sigma", formats)


def main() -> None:
    args = parse_args()
    equation_dir = resolve_equation_dir(args.root)
    histories = load_histories(equation_dir)
    rewards = selected_rewards(histories, args.rewards, args.no_fixed)
    formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]
    out_dir = args.out or equation_dir / "reward_sweep" / "history_plots"
    color_map_path = args.color_map or out_dir / "reward_colors.json"
    color_map = load_color_map(color_map_path, rewards)
    save_color_map(color_map_path, color_map)

    for phase in (None, "adam", "lbfgs"):
        plot_metric(
            histories,
            rewards,
            color_map,
            out_dir,
            formats,
            metric="equal_weight_total",
            ylabel="equal-weight loss",
            stem="equal_loss",
            phase=phase,
            log_y=True,
        )
        plot_metric(
            histories,
            rewards,
            color_map,
            out_dir,
            formats,
            metric="weighted_total",
            ylabel="weighted loss",
            stem="weighted_loss",
            phase=phase,
            log_y=True,
        )
        plot_metric(
            histories,
            rewards,
            color_map,
            out_dir,
            formats,
            metric="relative_l2",
            ylabel="relative L2",
            stem="relative_l2",
            phase=phase,
            log_y=True,
        )
        plot_components(histories, rewards, color_map, out_dir, formats, phase=phase)
        plot_weights(histories, rewards, color_map, out_dir, formats, phase=phase)

    plot_agent_reward(
        histories,
        rewards,
        color_map,
        out_dir,
        formats,
        window=args.reward_window,
    )
    plot_agent_sigma(histories, rewards, color_map, out_dir, formats)
    print(f"Plots: {out_dir}")
    print(f"Colors: {color_map_path}")


if __name__ == "__main__":
    main()
