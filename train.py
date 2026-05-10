from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pinn_accel.config import ExperimentConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact PINN experiment.")
    parser.add_argument("--config", type=Path, help="JSON config path.")
    parser.add_argument("--equation", help="Equation name from the registry.")
    parser.add_argument("--controllers", help="Comma-separated controllers.")
    parser.add_argument("--steps", type=int, help="Training steps.")
    parser.add_argument(
        "--optimizer-mode",
        choices=["adam", "adam_lbfgs", "lbfgs"],
        help="PINN optimizer schedule.",
    )
    parser.add_argument("--adam-steps", type=int, help="Adam phase steps.")
    parser.add_argument("--lbfgs-steps", type=int, help="L-BFGS phase steps.")
    parser.add_argument("--device", help="auto, cpu, cuda, cuda:0, mps.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--out", dest="output_dir", help="Artifacts directory.")
    parser.add_argument("--reward", help="Reward for agent controllers.")
    parser.add_argument("--compile", dest="compile_model", action="store_true")
    parser.add_argument("--no-plots", dest="save_plots", action="store_false")
    parser.set_defaults(save_plots=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig.from_file(args.config) if args.config else ExperimentConfig()
    cfg = cfg.with_cli_overrides(
        equation=args.equation,
        controllers=args.controllers,
        steps=args.steps,
        device=args.device,
        seed=args.seed,
        output_dir=args.output_dir,
        reward=args.reward,
        optimizer_mode=args.optimizer_mode,
        adam_steps=args.adam_steps,
        lbfgs_steps=args.lbfgs_steps,
        compile_model=args.compile_model,
        save_plots=args.save_plots,
    )
    try:
        from pinn_accel.experiment import run_experiment
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing runtime dependency. Run: python3 -m venv .venv && "
            ".venv/bin/pip install -e ."
        ) from exc

    run_dir = run_experiment(cfg)
    print(f"Artifacts: {run_dir}")


if __name__ == "__main__":
    main()
