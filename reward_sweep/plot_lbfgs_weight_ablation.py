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


RUNS = {
    "A_fixed_equal": {
        "dir": "a_fixed_equal",
        "label": "A fixed Adam -> equal L-BFGS",
        "short": "A fixed",
        "color": "#4d4d4d",
        "linestyle": "-",
    },
    "agent_adam_pre_lbfgs": {
        "dir": "agent_adam_pre_lbfgs",
        "label": "agent Adam only",
        "short": "agent Adam",
        "color": "#1f77b4",
        "linestyle": "--",
    },
    "B_agent_learned_lbfgs": {
        "dir": "b_agent_learned_lbfgs",
        "label": "B agent Adam -> learned-weight L-BFGS",
        "short": "B learned",
        "color": "#1f77b4",
        "linestyle": "-",
    },
    "C_agent_equal_lbfgs": {
        "dir": "c_agent_equal_lbfgs",
        "label": "C agent Adam -> equal L-BFGS",
        "short": "C equal",
        "color": "#2ca02c",
        "linestyle": "-",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Adam-only and L-BFGS ablation plots from saved histories."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Ablation run folder or equation folder containing A/B/C history.json files.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output plot directory. Default: <equation>/lbfgs_weight_ablation/plots.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats, e.g. pdf,png.",
    )
    return parser.parse_args()


def resolve_equation_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "lbfgs_weight_ablation_summary.json").is_file():
        return root
    direct = [
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "lbfgs_weight_ablation_summary.json").is_file()
    ]
    if len(direct) == 1:
        return direct[0]
    histories = [root / spec["dir"] / "history.json" for spec in RUNS.values()]
    if any(path.is_file() for path in histories):
        return root
    raise FileNotFoundError(f"Could not find ablation histories under {root}")


def load_history(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def load_histories(equation_dir: Path) -> dict[str, dict[str, Any]]:
    histories: dict[str, dict[str, Any]] = {}
    for name, spec in RUNS.items():
        path = equation_dir / spec["dir"] / "history.json"
        if path.is_file():
            histories[name] = load_history(path)
    required = {"A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"}
    missing = sorted(required - histories.keys())
    if missing:
        raise FileNotFoundError(f"Missing histories for: {', '.join(missing)}")
    return histories


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


def adam_stop(history: dict[str, Any]) -> int:
    start = phase_start(history, "lbfgs")
    if start is None:
        return len(history["equal_weight_total"])
    return start


def save(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_phase_boundary(history: dict[str, Any]) -> None:
    start = phase_start(history, "lbfgs")
    if start is None:
        return
    plt.axvline(start + 1, color="black", linewidth=1.0, alpha=0.35)
    ymin, ymax = plt.ylim()
    plt.text(
        start + 1,
        ymax,
        " L-BFGS",
        va="top",
        ha="left",
        fontsize=9,
        color="black",
        alpha=0.7,
    )


def plot_metric(
    histories: dict[str, dict[str, Any]],
    keys: list[str],
    metric: str,
    ylabel: str,
    out_dir: Path,
    stem: str,
    formats: list[str],
    *,
    log_y: bool = True,
    adam_only: bool = False,
    lbfgs_only: bool = False,
) -> None:
    fig = plt.figure(figsize=(8, 4.5))
    for key in keys:
        history = histories[key]
        values = series(history.get(metric, []))
        if not values.size:
            continue
        if adam_only:
            stop = adam_stop(history)
            values = values[:stop]
            x = np.arange(1, len(values) + 1)
        elif lbfgs_only:
            start = phase_start(history, "lbfgs")
            if start is None:
                continue
            values = values[start:]
            x = np.arange(1, len(values) + 1)
        else:
            x = np.arange(1, len(values) + 1)
        mask = np.isfinite(values)
        if not np.any(mask):
            continue
        plot = plt.semilogy if log_y else plt.plot
        plot(
            x[mask],
            values[mask],
            label=RUNS[key]["label"],
            color=RUNS[key]["color"],
            linestyle=RUNS[key]["linestyle"],
            linewidth=2.0,
        )
    if not adam_only and not lbfgs_only:
        draw_phase_boundary(histories[keys[0]])
    plt.xlabel("Adam step" if adam_only else "L-BFGS outer step" if lbfgs_only else "step")
    plt.ylabel(ylabel)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(fontsize=9)
    save(fig, out_dir, stem, formats)


def plot_adam_components(
    histories: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> None:
    keys = ["A_fixed_equal", "agent_adam_pre_lbfgs"]
    names = list(histories["A_fixed_equal"]["component_names"])
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 3.8))
    axes = np.atleast_1d(axes)
    for axis, component in zip(axes, names):
        for key in keys:
            history = histories[key]
            stop = adam_stop(history)
            values = series(history["components"][component])[:stop]
            x = np.arange(1, len(values) + 1)
            mask = np.isfinite(values)
            axis.semilogy(
                x[mask],
                values[mask],
                label=RUNS[key]["short"],
                color=RUNS[key]["color"],
                linestyle=RUNS[key]["linestyle"],
                linewidth=2.0,
            )
        axis.set_title(component)
        axis.set_xlabel("Adam step")
        axis.grid(True, which="both", ls="--", alpha=0.3)
    axes[0].set_ylabel("component loss")
    axes[0].legend(fontsize=8)
    save(fig, out_dir, "adam_component_losses", formats)


def plot_weights(
    histories: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> None:
    key = "agent_adam_pre_lbfgs" if "agent_adam_pre_lbfgs" in histories else "B_agent_learned_lbfgs"
    history = histories[key]
    names = list(history["component_names"])
    weights = np.asarray(history["weights"], dtype=np.float64)
    stop = adam_stop(history)
    fig = plt.figure(figsize=(8, 4.5))
    for idx, name in enumerate(names):
        plt.plot(
            np.arange(1, stop + 1),
            weights[:stop, idx],
            label=f"w_{name}",
            linewidth=2.0,
        )
    plt.xlabel("Adam step")
    plt.ylabel("agent weight")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    save(fig, out_dir, "adam_agent_weights", formats)


def plot_reward(
    histories: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> None:
    key = "agent_adam_pre_lbfgs" if "agent_adam_pre_lbfgs" in histories else "B_agent_learned_lbfgs"
    history = histories[key]
    rewards = series(history.get("agent_reward", []))
    stop = adam_stop(history)
    rewards = rewards[:stop]
    mask = np.isfinite(rewards)
    if not np.any(mask):
        return
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(
        np.arange(1, len(rewards) + 1)[mask],
        rewards[mask],
        color=RUNS[key]["color"],
        linewidth=1.5,
    )
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.35)
    plt.xlabel("Adam step")
    plt.ylabel("agent reward")
    plt.grid(True, ls="--", alpha=0.3)
    save(fig, out_dir, "adam_agent_reward", formats)


def plot_pre_post_bars(
    histories: dict[str, dict[str, Any]],
    out_dir: Path,
    formats: list[str],
) -> None:
    labels = ["A fixed", "B learned", "C equal"]
    keys = ["A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"]
    pre = []
    post = []
    for key in keys:
        history = histories[key]
        stop = adam_stop(history)
        l2 = series(history.get("relative_l2", []))
        finite_pre = l2[:stop][np.isfinite(l2[:stop])]
        finite_post = l2[np.isfinite(l2)]
        pre.append(float(finite_pre[-1]) if finite_pre.size else math.nan)
        post.append(float(finite_post[-1]) if finite_post.size else math.nan)
    x = np.arange(len(keys))
    width = 0.36
    fig = plt.figure(figsize=(7.5, 4.5))
    plt.bar(x - width / 2, pre, width, label="pre L-BFGS", color="#9ecae1")
    plt.bar(x + width / 2, post, width, label="post L-BFGS", color="#3182bd")
    plt.xticks(x, labels)
    plt.ylabel("relative L2")
    plt.grid(True, axis="y", ls="--", alpha=0.3)
    plt.legend()
    save(fig, out_dir, "pre_post_relative_l2_bars", formats)


def main() -> None:
    args = parse_args()
    equation_dir = resolve_equation_dir(args.root)
    histories = load_histories(equation_dir)
    formats = [item.strip().lstrip(".") for item in args.formats.split(",") if item.strip()]
    out_dir = args.out or equation_dir / "lbfgs_weight_ablation" / "plots"

    plot_metric(
        histories,
        ["A_fixed_equal", "agent_adam_pre_lbfgs"],
        "relative_l2",
        "relative L2",
        out_dir,
        "adam_relative_l2",
        formats,
        adam_only=True,
    )
    plot_metric(
        histories,
        ["A_fixed_equal", "agent_adam_pre_lbfgs"],
        "equal_weight_total",
        "equal-weight loss",
        out_dir,
        "adam_equal_loss",
        formats,
        adam_only=True,
    )
    plot_adam_components(histories, out_dir, formats)
    plot_weights(histories, out_dir, formats)
    plot_reward(histories, out_dir, formats)
    plot_metric(
        histories,
        ["A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"],
        "relative_l2",
        "relative L2",
        out_dir,
        "full_relative_l2",
        formats,
    )
    plot_metric(
        histories,
        ["A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"],
        "equal_weight_total",
        "equal-weight loss",
        out_dir,
        "full_equal_loss",
        formats,
    )
    plot_metric(
        histories,
        ["A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"],
        "relative_l2",
        "relative L2",
        out_dir,
        "lbfgs_relative_l2_zoom",
        formats,
        lbfgs_only=True,
    )
    plot_metric(
        histories,
        ["A_fixed_equal", "B_agent_learned_lbfgs", "C_agent_equal_lbfgs"],
        "equal_weight_total",
        "equal-weight loss",
        out_dir,
        "lbfgs_equal_loss_zoom",
        formats,
        lbfgs_only=True,
    )
    plot_pre_post_bars(histories, out_dir, formats)
    print(f"Plots: {out_dir}")


if __name__ == "__main__":
    main()
