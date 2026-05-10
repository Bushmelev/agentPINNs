from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .equations.base import EquationSpec


METHOD_COLORS = {
    "fixed": "#4d4d4d",
    "agent": "#1f77b4",
    "tiny_loss_weight": "#1f77b4",
    "tinylossweight": "#1f77b4",
    "softadapt": "#ff7f0e",
    "relobralo": "#2ca02c",
    "gradnorm": "#d62728",
}

METHOD_LABELS = {
    "fixed": "fixed",
    "agent": "agent",
    "tiny_loss_weight": "agent",
    "tinylossweight": "agent",
    "softadapt": "SoftAdapt",
    "relobralo": "ReLoBRaLo",
    "gradnorm": "GradNorm",
}


def _method_key(label: str) -> str:
    normalized = label.lower().replace("-", "_").replace(" ", "_")
    compact = normalized.replace("_", "")
    if "tinylossweight" in compact or "agent" in normalized:
        return "agent"
    for key in ("fixed", "softadapt", "relobralo", "gradnorm"):
        if key in compact:
            return key
    return normalized


def _method_color(label: str) -> str | None:
    key = _method_key(label)
    return METHOD_COLORS.get(key) or METHOD_COLORS.get(label.lower())


def _method_label(label: str) -> str:
    key = _method_key(label)
    return METHOD_LABELS.get(key, key)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_history_plots(history: Mapping, plot_dir: Path) -> None:
    steps = np.arange(1, len(history["equal_weight_total"]) + 1)
    method_color = _method_color(str(history.get("controller", "")))

    fig = plt.figure(figsize=(7, 4))
    plt.semilogy(steps, history["equal_weight_total"], label="equal", color="#7f7f7f")
    plt.semilogy(
        steps,
        history["weighted_total"],
        label="weighted",
        color=method_color,
    )
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save(fig, plot_dir / "loss_total.png")

    relative_l2 = history.get("relative_l2")
    if relative_l2 is not None and any(value is not None for value in relative_l2):
        values = np.array(
            [np.nan if value is None else float(value) for value in relative_l2],
            dtype=np.float64,
        )
        mask = np.isfinite(values)
        fig = plt.figure(figsize=(7, 4))
        plt.semilogy(
            steps[mask],
            values[mask],
            color=method_color,
        )
        plt.xlabel("step")
        plt.ylabel("relative L2 error")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        _save(fig, plot_dir / "relative_l2.png")

    fig = plt.figure(figsize=(7, 4))
    for name in history["component_names"]:
        plt.semilogy(steps, history["components"][name], label=name)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save(fig, plot_dir / "loss_components.png")

    weights = np.asarray(history["weights"], dtype=np.float64)
    fig = plt.figure(figsize=(7, 4))
    for idx, name in enumerate(history["component_names"]):
        plt.plot(steps, weights[:, idx], label=f"w_{name}")
    plt.xlabel("step")
    plt.ylabel("weight")
    plt.grid(True, ls="--", alpha=0.3)
    plt.legend()
    _save(fig, plot_dir / "weights.png")

    rewards = history.get("agent_reward")
    if rewards is not None and any(value is not None for value in rewards):
        reward_values = np.array(
            [np.nan if value is None else float(value) for value in rewards],
            dtype=np.float64,
        )
        mask = np.isfinite(reward_values)
        fig = plt.figure(figsize=(7, 4))
        plt.plot(
            steps[mask],
            reward_values[mask],
            marker="o",
            markersize=3,
            color=method_color,
        )
        plt.axhline(0.0, color="black", linewidth=1, alpha=0.35)
        plt.xlabel("step")
        plt.ylabel("reward")
        plt.grid(True, ls="--", alpha=0.3)
        _save(fig, plot_dir / "agent_reward.png")


def save_comparison_plots(histories: Mapping[str, Mapping], plot_dir: Path) -> None:
    if not histories:
        return
    fig = plt.figure(figsize=(7, 4))
    for label, history in histories.items():
        plt.semilogy(
            history["equal_weight_total"],
            label=_method_label(label),
            color=_method_color(label),
        )
    plt.xlabel("step")
    plt.ylabel("equal-weight loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save(fig, plot_dir / "comparison_equal_loss.png")

    fig = plt.figure(figsize=(7, 4))
    for label, history in histories.items():
        plt.semilogy(
            history["weighted_total"],
            label=_method_label(label),
            color=_method_color(label),
        )
    plt.xlabel("step")
    plt.ylabel("weighted loss")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    _save(fig, plot_dir / "comparison_weighted_loss.png")

    if any(history.get("relative_l2") for history in histories.values()):
        fig = plt.figure(figsize=(7, 4))
        for label, history in histories.items():
            values = np.array(
                [
                    np.nan if value is None else float(value)
                    for value in history.get("relative_l2", [])
                ],
                dtype=np.float64,
            )
            mask = np.isfinite(values)
            if not np.any(mask):
                continue
            plt.semilogy(
                np.arange(1, len(values) + 1)[mask],
                values[mask],
                label=_method_label(label),
                color=_method_color(label),
            )
        plt.xlabel("step")
        plt.ylabel("relative L2 error")
        plt.grid(True, which="both", ls="--", alpha=0.3)
        plt.legend()
        _save(fig, plot_dir / "comparison_relative_l2.png")

    component_names = list(next(iter(histories.values()))["component_names"])
    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4 * len(component_names), 3.5),
    )
    axes = np.atleast_1d(axes)
    for axis, component in zip(axes, component_names):
        for label, history in histories.items():
            axis.semilogy(
                history["components"][component],
                label=_method_label(label),
                color=_method_color(label),
            )
        axis.set_title(component)
        axis.set_xlabel("step")
        axis.grid(True, which="both", ls="--", alpha=0.3)
    axes[0].set_ylabel("loss")
    axes[0].legend()
    _save(fig, plot_dir / "comparison_components.png")

    fig, axes = plt.subplots(
        1,
        len(component_names),
        figsize=(4 * len(component_names), 3.5),
    )
    axes = np.atleast_1d(axes)
    for idx, (axis, component) in enumerate(zip(axes, component_names)):
        for label, history in histories.items():
            weights = np.asarray(history["weights"], dtype=np.float64)
            axis.plot(
                weights[:, idx],
                label=_method_label(label),
                color=_method_color(label),
            )
        axis.set_title(f"w_{component}")
        axis.set_xlabel("step")
        axis.grid(True, ls="--", alpha=0.3)
    axes[0].set_ylabel("weight")
    axes[0].legend()
    _save(fig, plot_dir / "comparison_weights.png")


def evaluate_grid(
    model: nn.Module,
    spec: EquationSpec,
    device: torch.device,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = torch.linspace(spec.x_min, spec.x_max, n, device=device)
    t = torch.linspace(spec.t_min, spec.t_max, n, device=device)
    grid_x, grid_t = torch.meshgrid(x, t, indexing="ij")
    xt = torch.stack([grid_x.reshape(-1), grid_t.reshape(-1)], dim=1)
    with torch.no_grad():
        u = model(xt).reshape(n, n).detach().cpu().numpy()
    return x.detach().cpu().numpy(), t.detach().cpu().numpy(), u


def save_solution_plot(
    model: nn.Module,
    spec: EquationSpec,
    device: torch.device,
    plot_dir: Path,
    n: int = 120,
) -> None:
    x, t, u = evaluate_grid(model, spec, device, n)
    fig = plt.figure(figsize=(7, 4))
    plt.imshow(
        u.T,
        extent=[x.min(), x.max(), t.min(), t.max()],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )
    plt.colorbar(label="u(x,t)")
    plt.xlabel("x")
    plt.ylabel("t")
    _save(fig, plot_dir / "solution.png")
