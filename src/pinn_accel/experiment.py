from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .config import ExperimentConfig
from .controllers import controller_needs_baseline, make_controller
from .equations import get_equation
from .equations.base import EquationSpec
from .plots import save_comparison_plots, save_history_plots, save_solution_plot
from .settings import configure_torch, resolve_device
from .training import TrainResult, train_one


def _controller_params(cfg: ExperimentConfig, name: str) -> dict[str, Any]:
    return dict(cfg.controller_params.get(name, {}))


def _save_result(
    *,
    store: ArtifactStore,
    cfg: ExperimentConfig,
    result: TrainResult,
    spec: EquationSpec,
    label: str,
    device,
) -> None:
    method_dir = store.method_dir(spec.name, label)
    store.save_history(spec.name, label, result.history)
    if result.history.get("batch_info") is not None:
        store.save_json(method_dir / "batch_info.json", result.history["batch_info"])
    store.save_checkpoint(
        spec.name,
        label,
        {
            "equation": spec.name,
            "controller": label,
            "model_config": (
                cfg.model.to_dict()
                if hasattr(cfg.model, "to_dict")
                else cfg.model.__dict__
            ),
            "training_config": cfg.training.__dict__,
            "state_dict": result.model.state_dict(),
            "history": result.history,
        },
    )
    if cfg.save_plots:
        save_history_plots(result.history, method_dir / "plots")
        save_solution_plot(
            result.model,
            spec,
            device,
            method_dir / "plots",
            n=cfg.plot_grid,
        )


def _last_finite(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return finite[-1] if finite else None


def run_experiment(cfg: ExperimentConfig) -> Path:
    configure_torch()
    device = resolve_device(cfg.device)
    spec = get_equation(cfg.equation, **cfg.equation_params)
    store = ArtifactStore.create(cfg.output_dir)
    store.save_json(store.root / "config.json", cfg.to_dict())
    print(f"Device: {device}")
    print(f"Run: {store.root}")

    results: dict[str, TrainResult] = {}
    histories_for_compare: dict[str, dict] = {}
    baseline_history: dict[str, Any] | None = None

    controllers = list(cfg.controllers)
    needs_baseline = any(
        controller_needs_baseline(name, _controller_params(cfg, name))
        for name in controllers
    )
    if needs_baseline:
        controllers = ["fixed", *[name for name in controllers if name != "fixed"]]

    for name in controllers:
        params = _controller_params(cfg, name)
        controller = make_controller(
            name,
            params,
            update_interval=cfg.training.agent_update_interval,
            warmup_steps=cfg.training.agent_warmup_steps,
        )
        if controller_needs_baseline(name, params) and baseline_history is None:
            raise RuntimeError("Baseline history is required but fixed controller has not run")

        result = train_one(
            spec=spec,
            model_cfg=cfg.model,
            train_cfg=cfg.training,
            controller=controller,
            device=device,
            seed=cfg.seed,
            baseline_history=baseline_history,
        )
        label = name
        results[label] = result
        histories_for_compare[label] = result.history
        _save_result(
            store=store,
            cfg=cfg,
            result=result,
            spec=spec,
            label=label,
            device=device,
        )
        print(f"[{spec.name}/{label}] elapsed={result.elapsed_sec:.1f}s")
        if name == "fixed":
            baseline_history = result.history

    if cfg.save_plots:
        compare_dir = store.equation_dir(spec.name) / "comparison" / "plots"
        save_comparison_plots(histories_for_compare, compare_dir)

    summary = {
        label: {
            "elapsed_sec": result.elapsed_sec,
            "final_equal_weight_total": result.history["equal_weight_total"][-1],
            "final_weighted_total": result.history["weighted_total"][-1],
            "final_relative_l2": _last_finite(result.history.get("relative_l2", [])),
            "final_weights": result.history["weights"][-1],
        }
        for label, result in results.items()
    }
    store.save_json(store.equation_dir(spec.name) / "summary.json", summary)
    print(f"Summary: {store.equation_dir(spec.name) / 'summary.json'}")
    return store.root
