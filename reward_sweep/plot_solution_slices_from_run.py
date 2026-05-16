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

from pinn_accel.checkpoints import load_checkpoint_payload  # noqa: E402
from pinn_accel.config import ExperimentConfig  # noqa: E402
from pinn_accel.equations import get_equation  # noqa: E402
from pinn_accel.equations.base import EquationSpec  # noqa: E402
from pinn_accel.models import MLP  # noqa: E402
from pinn_accel.settings import configure_torch, resolve_device  # noqa: E402


DEFAULT_COLORS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot solution time slices from saved reward-sweep checkpoints."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Sweep root or equation folder, e.g. .../<timestamp> or .../<timestamp>/burgers.",
    )
    parser.add_argument(
        "--rewards",
        required=True,
        help="Comma-separated methods to plot. Use fixed and reward names, e.g. fixed,component_balance_penalty.",
    )
    parser.add_argument(
        "--times",
        default=None,
        help="Comma-separated time values. Default: solution_slice_times from config.",
    )
    parser.add_argument(
        "--equation",
        help="Equation folder/config name if root contains multiple equations.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory. Default: <equation>/reward_sweep/solution_slices_selected.",
    )
    parser.add_argument(
        "--formats",
        default="pdf,png",
        help="Comma-separated output formats, e.g. pdf,png.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, mps.",
    )
    parser.add_argument(
        "--color-map",
        type=Path,
        help="Optional reward_colors.json from history_plots.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=65536,
        help="Prediction chunk size.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def resolve_equation_dir(root: Path, equation: str | None) -> Path:
    root = root.resolve()
    if (root / "reward_sweep_config.json").is_file():
        if equation is None:
            config = load_json(root / "reward_sweep_config.json")
            equation = str(config.get("equation", ""))
        candidate = root / str(equation)
        if not candidate.is_dir():
            raise FileNotFoundError(f"Equation folder not found: {candidate}")
        return candidate

    if (root.parent / "reward_sweep_config.json").is_file():
        if equation is not None and root.name != equation:
            candidate = root.parent / equation
            if not candidate.is_dir():
                raise FileNotFoundError(f"Equation folder not found: {candidate}")
            return candidate
        return root

    candidates = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and list(path.glob("*/checkpoint.pt"))
    )
    if equation is not None:
        candidate = root / equation
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Equation folder not found: {candidate}")
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(path.name for path in candidates)
    raise ValueError(f"Cannot infer equation folder. Pass --equation. Found: {names}")


def config_for_equation(equation_dir: Path) -> ExperimentConfig:
    for path in (
        equation_dir.parent / "reward_sweep_config.json",
        equation_dir.parent / "config.json",
    ):
        if path.is_file():
            return ExperimentConfig.from_dict(load_json(path))
    raise FileNotFoundError(
        f"Could not find reward_sweep_config.json or config.json near {equation_dir}"
    )


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_times(value: str | None, cfg: ExperimentConfig) -> list[float]:
    if value is None:
        return [float(item) for item in cfg.solution_slice_times]
    return [float(item) for item in parse_csv(value)]


def method_dir(equation_dir: Path, name: str) -> Path:
    if name == "fixed":
        return equation_dir / "fixed"
    direct = equation_dir / name
    if direct.is_dir():
        return direct
    agent = equation_dir / f"agent_{name}"
    if agent.is_dir():
        return agent
    raise FileNotFoundError(
        f"Could not find method folder for {name!r} under {equation_dir}"
    )


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    payload = load_checkpoint_payload(checkpoint_path, map_location=device)
    model_cfg = payload.get("model_config", {})
    if not isinstance(model_cfg, dict):
        raise ValueError(f"checkpoint model_config must be a dict: {checkpoint_path}")
    layers = model_cfg.get("layers")
    activation = str(model_cfg.get("activation", "tanh"))
    if not isinstance(layers, list):
        raise ValueError(f"checkpoint model_config.layers is missing: {checkpoint_path}")
    model = MLP(layers, activation=activation).to(device)
    state_dict = payload.get("model_state_dict") or payload.get("state_dict")
    if state_dict is None:
        raise ValueError(f"checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def reference_grid(spec: EquationSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if spec.reference_solver is None:
        return None
    x, t, u = spec.solve_reference()
    if u.shape == (len(t), len(x)):
        u = u.T
    if u.shape != (len(x), len(t)):
        raise ValueError(f"Reference solution shape must be {(len(x), len(t))}, got {u.shape}")
    return x, t, u


def predict_slice(
    model: torch.nn.Module,
    x: np.ndarray,
    t_value: float,
    device: torch.device,
    chunk_size: int,
) -> np.ndarray:
    xt_np = np.stack([x, np.full_like(x, t_value)], axis=1)
    xt = torch.tensor(xt_np, dtype=torch.float32, device=device)
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, xt.shape[0], chunk_size):
            stop = min(start + chunk_size, xt.shape[0])
            values.append(model(xt[start:stop]).reshape(-1).detach().cpu().numpy())
    return np.concatenate(values)


def load_color_map(path: Path | None, names: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {"fixed": "#111111"}
    if path is not None:
        loaded = load_json(path)
        colors.update({str(key): str(value) for key, value in loaded.items()})
    for idx, name in enumerate(name for name in names if name not in colors):
        colors[name] = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]
    return colors


def plot_slices(
    *,
    models: dict[str, torch.nn.Module],
    spec: EquationSpec,
    times: list[float],
    device: torch.device,
    out_dir: Path,
    formats: list[str],
    colors: dict[str, str],
    chunk_size: int,
) -> None:
    reference = reference_grid(spec)
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
        raise ValueError(
            f"No requested times are inside [{spec.t_min}, {spec.t_max}]: {times}"
        )

    ncols = min(2, len(valid_times))
    nrows = int(np.ceil(len(valid_times) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.2 * ncols + 2.8, 3.8 * nrows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.42, wspace=0.25, right=0.78)
    axes_flat = axes.reshape(-1)
    handles = []
    labels = []

    for plot_idx, (axis, requested_time) in enumerate(zip(axes_flat, valid_times)):
        plot_time = requested_time
        if t_ref is not None and u_ref is not None:
            time_index = int(np.argmin(np.abs(t_ref - requested_time)))
            plot_time = float(t_ref[time_index])
            (line,) = axis.plot(
                x,
                u_ref[:, time_index],
                label="reference",
                color="#000000",
                linewidth=2.0,
                linestyle="--",
            )
            if "reference" not in labels:
                handles.append(line)
                labels.append("reference")

        for name, model in models.items():
            prediction = predict_slice(model, x, plot_time, device, chunk_size)
            (line,) = axis.plot(
                x,
                prediction,
                label=name,
                color=colors.get(name),
                linewidth=1.6 if name != "fixed" else 2.0,
                alpha=0.95 if name != "fixed" else 0.85,
            )
            if name not in labels:
                handles.append(line)
                labels.append(name)

        axis.set_title(f"t={plot_time:.3g}")
        if plot_idx // ncols == nrows - 1:
            axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.grid(True, ls="--", alpha=0.3)

    for axis in axes_flat[len(valid_times) :]:
        axis.axis("off")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.80, 0.5),
        frameon=False,
        fontsize=8,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(out_dir / f"solution_slices.{fmt}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    configure_torch()
    equation_dir = resolve_equation_dir(args.root, args.equation)
    cfg = config_for_equation(equation_dir)
    spec = get_equation(cfg.equation, **cfg.equation_params)
    device = resolve_device(args.device)
    names = parse_csv(args.rewards)
    if not names:
        raise SystemExit("--rewards must contain at least one method")
    times = parse_times(args.times, cfg)
    out_dir = args.out or equation_dir / "reward_sweep" / "solution_slices_selected"
    formats = parse_csv(args.formats)
    if not formats:
        formats = ["pdf"]

    models = {
        name: load_model(method_dir(equation_dir, name) / "checkpoint.pt", device)
        for name in names
    }
    colors = load_color_map(args.color_map, names)
    plot_slices(
        models=models,
        spec=spec,
        times=times,
        device=device,
        out_dir=out_dir,
        formats=formats,
        colors=colors,
        chunk_size=max(1, int(args.chunk_size)),
    )
    print(f"Saved solution slices to {out_dir}")


if __name__ == "__main__":
    main()
