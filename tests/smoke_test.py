from __future__ import annotations

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_accel.config import ExperimentConfig, ModelConfig, TrainingConfig
from pinn_accel.experiment import run_experiment


def main() -> None:
    out_dir = ROOT / "artifacts_smoke"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    cfg = ExperimentConfig(
        equation="heat",
        controllers=["fixed"],
        output_dir=str(out_dir),
        save_plots=False,
        model=ModelConfig(layers=[2, 8, 8, 1], activation="tanh"),
        training=TrainingConfig(
            steps=3,
            batch_sizes={"pde": 16, "ic": 8, "bc": 8},
            pool_sizes={},
            log_every=1,
        ),
    )
    run_dir = run_experiment(cfg)
    assert (run_dir / "heat" / "fixed" / "history.json").exists()


if __name__ == "__main__":
    main()
