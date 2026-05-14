from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pinn_accel.artifacts import ArtifactStore, to_jsonable  # noqa: E402
from pinn_accel.checkpoints import build_result_checkpoint  # noqa: E402
from pinn_accel.config import ExperimentConfig  # noqa: E402
from pinn_accel.controllers import make_controller  # noqa: E402
from pinn_accel.equations import get_equation  # noqa: E402
from pinn_accel.equations.base import EquationSpec  # noqa: E402
from pinn_accel.losses import LossEvaluator  # noqa: E402
from pinn_accel.optim import make_lbfgs_optimizer  # noqa: E402
from pinn_accel.plots import (  # noqa: E402
    save_comparison_plots,
    save_history_plots,
    save_solution_plot,
    save_solution_slice_comparison,
)
from pinn_accel.settings import configure_torch, resolve_device  # noqa: E402
from pinn_accel.training import (  # noqa: E402
    TrainResult,
    _build_relative_l2_metric,
    _history_values,
    train_one,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run A/B/C L-BFGS weight ablation with one shared agent Adam pretrain."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config path. If omitted, uses the built-in synthetic Burgers config.",
    )
    parser.add_argument("--controller", default="tiny_loss_weight")
    parser.add_argument("--reward", default="component_relative_improvement")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "artifacts" / "lbfgs_weight_ablation")
    parser.add_argument("--steps", type=int, help="Total Adam + L-BFGS steps.")
    parser.add_argument("--adam-steps", type=int, default=4000)
    parser.add_argument("--lbfgs-steps", type=int, default=200)
    parser.add_argument("--lbfgs-max-iter", type=int, default=20)
    parser.add_argument("--agent-update-interval", type=int)
    parser.add_argument("--agent-warmup-steps", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=None)
    return parser.parse_args()


def config_with_overrides(args: argparse.Namespace) -> ExperimentConfig:
    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    cfg = cfg.with_cli_overrides(
        controllers=f"fixed,{args.controller}",
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        output_dir=str(args.out),
        reward=args.reward,
        optimizer_mode="adam_lbfgs",
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        lbfgs_max_iter=args.lbfgs_max_iter,
        agent_update_interval=args.agent_update_interval,
        agent_warmup_steps=args.agent_warmup_steps,
        save_plots=args.save_plots,
    )
    data = cfg.to_dict()
    data["controllers"] = ["fixed", args.controller]
    data["training"]["steps"] = int(args.adam_steps + args.lbfgs_steps)
    return ExperimentConfig.from_dict(data)


def clone_with_training(cfg: ExperimentConfig, **training_overrides: Any) -> ExperimentConfig:
    data = cfg.to_dict()
    data["training"].update(training_overrides)
    return ExperimentConfig.from_dict(data)


def controller_params(cfg: ExperimentConfig, name: str) -> dict[str, Any]:
    return dict(cfg.controller_params.get(name, {}))


def save_result(
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
        save_solution_plot(result.model, spec, device, method_dir / "plots", n=cfg.plot_grid)


def last_finite(values: list[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return finite[-1] if finite else None


def value_at(values: list[Any], index: int) -> float | None:
    if not values:
        return None
    index = min(max(index, 0), len(values) - 1)
    for idx in range(index, -1, -1):
        value = values[idx]
        if value is not None:
            return float(value)
    return None


def equal_objective(losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.full_like(losses, 1.0 / max(int(losses.numel()), 1))
    return torch.sum(weights * losses), weights.detach()


def branch_objective(
    *,
    losses: torch.Tensor,
    model: torch.nn.Module,
    controller,
    step: int,
    weight_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_mode == "equal":
        return equal_objective(losses)
    return controller.frozen_objective(losses, model, step)


def rescale_prefix_progress(history: dict[str, Any], total_steps: int, adam_steps: int) -> None:
    for idx in range(len(history.get("progress", []))):
        step = idx + 1
        history["progress"][idx] = step / float(max(total_steps, 1))
    for idx in range(len(history.get("agent_progress", []))):
        step = idx + 1
        history["agent_progress"][idx] = min(step / float(max(adam_steps, 1)), 1.0)


def append_lbfgs_step(
    *,
    history: dict[str, Any],
    component_names: list[str],
    raw_losses: np.ndarray,
    weights_np: np.ndarray,
    equal_total: float,
    weighted_total: float,
    relative_l2: float | None,
    lr: float,
    step: int,
    total_steps: int,
    lbfgs_closure_calls: int,
    controller,
) -> None:
    extras = controller.frozen_step_extras()
    history["equal_weight_total"].append(equal_total)
    history["weighted_total"].append(weighted_total)
    history["relative_l2"].append(relative_l2)
    for name, value in zip(component_names, raw_losses):
        history["components"][name].append(float(value))
    history["weights"].append(weights_np.tolist())
    history["lr"].append(float(lr))
    history["progress"].append(step / float(max(total_steps, 1)))
    history["agent_progress"].append(1.0)
    history["optimizer_phase"].append("lbfgs")
    history["weights_frozen"].append(True)
    history["agent_reward"].append(extras.get("agent_reward"))
    history["agent_sigma"].append(extras.get("agent_sigma"))
    history["agent_frozen"].append(True)
    history["lbfgs_closure_calls"].append(lbfgs_closure_calls)


def run_lbfgs_branch(
    *,
    spec: EquationSpec,
    model: torch.nn.Module,
    controller,
    train_cfg,
    device: torch.device,
    seed: int,
    prefix_history: dict[str, Any],
    weight_mode: str,
) -> TrainResult:
    start_time = time.time()
    history = copy.deepcopy(prefix_history)
    adam_steps = len(history["equal_weight_total"])
    total_steps = adam_steps + int(train_cfg.lbfgs_steps or 0)
    rescale_prefix_progress(history, total_steps, adam_steps)

    loss_evaluator = LossEvaluator(
        spec,
        batch_sizes=train_cfg.batch_sizes,
        pool_sizes=train_cfg.pool_sizes,
        device=device,
        seed=seed,
        full_batch=train_cfg.full_batch,
    )
    history["batch_info"] = loss_evaluator.batch_info
    component_names = loss_evaluator.component_names
    relative_l2_metric = (
        _build_relative_l2_metric(spec, device, train_cfg.relative_l2_chunk_size)
        if train_cfg.relative_l2_every > 0
        else None
    )
    optimizer = make_lbfgs_optimizer(
        model.parameters(),
        lr=train_cfg.lbfgs_lr,
        max_iter=train_cfg.lbfgs_max_iter,
        max_eval=train_cfg.lbfgs_max_eval,
        history_size=train_cfg.lbfgs_history_size,
        tolerance_grad=train_cfg.lbfgs_tolerance_grad,
        tolerance_change=train_cfg.lbfgs_tolerance_change,
        line_search_fn=train_cfg.lbfgs_line_search_fn,
    )

    for phase_step in range(1, int(train_cfg.lbfgs_steps or 0) + 1):
        step = adam_steps + phase_step
        batches = loss_evaluator.draw_batches()
        lbfgs_closure_calls = 0

        def closure() -> torch.Tensor:
            nonlocal lbfgs_closure_calls
            lbfgs_closure_calls += 1
            optimizer.zero_grad(set_to_none=True)
            closure_pack = loss_evaluator.compute(model, batches)
            closure_objective, _ = branch_objective(
                losses=closure_pack.values,
                model=model,
                controller=controller,
                step=step,
                weight_mode=weight_mode,
            )
            closure_objective.backward()
            return closure_objective

        optimizer.step(closure)
        loss_pack = loss_evaluator.compute(model, batches)
        objective, weights_t = branch_objective(
            losses=loss_pack.values,
            model=model,
            controller=controller,
            step=step,
            weight_mode=weight_mode,
        )
        raw_losses, weights_np, equal_total, weighted_total = _history_values(
            loss_pack,
            component_names,
            weights_t,
        )
        del objective, loss_pack, weights_t

        relative_l2 = None
        if relative_l2_metric is not None and (
            step % train_cfg.relative_l2_every == 0 or step == total_steps
        ):
            relative_l2 = relative_l2_metric(model)

        append_lbfgs_step(
            history=history,
            component_names=component_names,
            raw_losses=raw_losses,
            weights_np=weights_np,
            equal_total=equal_total,
            weighted_total=weighted_total,
            relative_l2=relative_l2,
            lr=float(optimizer.param_groups[0]["lr"]),
            step=step,
            total_steps=total_steps,
            lbfgs_closure_calls=lbfgs_closure_calls,
            controller=controller,
        )

    return TrainResult(
        model=model,
        history=history,
        controller=controller,
        elapsed_sec=time.time() - start_time,
    )


def summary_row(result: TrainResult, adam_steps: int) -> dict[str, Any]:
    history = result.history
    lbfgs_start_index = next(
        (
            idx
            for idx, phase in enumerate(history.get("optimizer_phase", []))
            if phase == "lbfgs"
        ),
        None,
    )
    return {
        "elapsed_sec": result.elapsed_sec,
        "pre_equal_weight_total": value_at(history["equal_weight_total"], adam_steps - 1),
        "final_equal_weight_total": history["equal_weight_total"][-1],
        "pre_weighted_total": value_at(history["weighted_total"], adam_steps - 1),
        "final_weighted_total": history["weighted_total"][-1],
        "pre_relative_l2": value_at(history.get("relative_l2", []), adam_steps - 1),
        "final_relative_l2": last_finite(history.get("relative_l2", [])),
        "pre_weights": history["weights"][adam_steps - 1],
        "lbfgs_start_weights": (
            None if lbfgs_start_index is None else history["weights"][lbfgs_start_index]
        ),
        "final_weights": history["weights"][-1],
        "lbfgs_closure_calls_total": int(
            sum(value or 0 for value in history.get("lbfgs_closure_calls", []))
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = config_with_overrides(args)
    configure_torch()
    device = resolve_device(cfg.device)
    spec = get_equation(cfg.equation, **cfg.equation_params)
    store = ArtifactStore.create(cfg.output_dir)
    store.save_json(store.root / "ablation_config.json", cfg.to_dict())
    store.save_json(
        store.root / "ablation_info.json",
        {
            "A_fixed_equal": "fixed Adam -> L-BFGS equal weights",
            "B_agent_learned_lbfgs": "agent Adam -> L-BFGS final agent weights",
            "C_agent_equal_lbfgs": "agent Adam -> L-BFGS equal weights",
        },
    )
    print(f"Device: {device}")
    print(f"Run: {store.root}")

    fixed_cfg = clone_with_training(cfg, lbfgs_weight_mode="equal")
    fixed_controller = make_controller(
        "fixed",
        controller_params(cfg, "fixed"),
        update_interval=cfg.training.agent_update_interval,
        warmup_steps=cfg.training.agent_warmup_steps,
    )
    fixed_result = train_one(
        spec=spec,
        model_cfg=cfg.model,
        train_cfg=fixed_cfg.training,
        controller=fixed_controller,
        device=device,
        seed=cfg.seed,
    )
    save_result(
        store=store,
        cfg=fixed_cfg,
        result=fixed_result,
        spec=spec,
        label="A_fixed_equal",
        device=device,
    )
    print(f"[{spec.name}/A_fixed_equal] elapsed={fixed_result.elapsed_sec:.1f}s")

    agent_adam_cfg = clone_with_training(
        cfg,
        steps=int(cfg.training.adam_steps or args.adam_steps),
        optimizer_mode="adam",
        adam_steps=int(cfg.training.adam_steps or args.adam_steps),
        lbfgs_steps=0,
    )
    agent_controller = make_controller(
        args.controller,
        controller_params(cfg, args.controller),
        update_interval=cfg.training.agent_update_interval,
        warmup_steps=cfg.training.agent_warmup_steps,
    )
    agent_adam_result = train_one(
        spec=spec,
        model_cfg=cfg.model,
        train_cfg=agent_adam_cfg.training,
        controller=agent_controller,
        device=device,
        seed=cfg.seed,
        baseline_history=fixed_result.history,
    )
    save_result(
        store=store,
        cfg=agent_adam_cfg,
        result=agent_adam_result,
        spec=spec,
        label="agent_adam_pre_lbfgs",
        device=device,
    )
    print(f"[{spec.name}/agent_adam_pre_lbfgs] elapsed={agent_adam_result.elapsed_sec:.1f}s")

    branch_cfg = clone_with_training(cfg, optimizer_mode="lbfgs")
    branch_specs = [
        ("B_agent_learned_lbfgs", "controller"),
        ("C_agent_equal_lbfgs", "equal"),
    ]
    branch_results: dict[str, TrainResult] = {}
    for label, weight_mode in branch_specs:
        branch_model = copy.deepcopy(agent_adam_result.model).to(device)
        branch_controller = copy.deepcopy(agent_adam_result.controller).to(device)
        branch_result = run_lbfgs_branch(
            spec=spec,
            model=branch_model,
            controller=branch_controller,
            train_cfg=clone_with_training(branch_cfg, lbfgs_weight_mode=weight_mode).training,
            device=device,
            seed=cfg.seed,
            prefix_history=agent_adam_result.history,
            weight_mode=weight_mode,
        )
        branch_results[label] = branch_result
        save_result(
            store=store,
            cfg=clone_with_training(branch_cfg, lbfgs_weight_mode=weight_mode),
            result=branch_result,
            spec=spec,
            label=label,
            device=device,
        )
        print(f"[{spec.name}/{label}] elapsed={branch_result.elapsed_sec:.1f}s")

    histories_for_compare = {
        "A_fixed_equal": fixed_result.history,
        "B_agent_learned_lbfgs": branch_results["B_agent_learned_lbfgs"].history,
        "C_agent_equal_lbfgs": branch_results["C_agent_equal_lbfgs"].history,
    }
    if cfg.save_plots:
        compare_dir = store.equation_dir(spec.name) / "comparison" / "plots"
        save_comparison_plots(histories_for_compare, compare_dir)
        save_solution_slice_comparison(
            {
                "A_fixed_equal": fixed_result.model,
                "B_agent_learned_lbfgs": branch_results["B_agent_learned_lbfgs"].model,
                "C_agent_equal_lbfgs": branch_results["C_agent_equal_lbfgs"].model,
            },
            spec,
            device,
            compare_dir,
            times=cfg.solution_slice_times,
        )

    adam_steps = int(cfg.training.adam_steps or args.adam_steps)
    summary = {
        "A_fixed_equal": summary_row(fixed_result, adam_steps),
        "B_agent_learned_lbfgs": summary_row(
            branch_results["B_agent_learned_lbfgs"],
            adam_steps,
        ),
        "C_agent_equal_lbfgs": summary_row(
            branch_results["C_agent_equal_lbfgs"],
            adam_steps,
        ),
    }
    summary_path = store.equation_dir(spec.name) / "lbfgs_weight_ablation_summary.json"
    store.save_json(summary_path, summary)
    print(f"Summary: {summary_path}")
    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
