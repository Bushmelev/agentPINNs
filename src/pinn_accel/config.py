from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


AGENT_CONTROLLER_NAMES = {"tiny_loss_weight", "tiny", "tiny_rl", "linear_rl"}


@dataclass
class ModelConfig:
    layers: list[int] = field(default_factory=lambda: [2, 40, 40, 40, 40, 1])
    activation: str = "tanh"


@dataclass
class TrainingConfig:
    steps: int = 1000
    full_batch: bool = True
    batch_sizes: dict[str, int] = field(
        default_factory=lambda: {"pde": 2048, "ic": 256, "bc": 256},
    )
    pool_sizes: dict[str, int] = field(default_factory=dict)
    optimizer_mode: str = "adam"
    optimizer: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: str = "constant"
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    adam_steps: int | None = None
    lbfgs_steps: int | None = None
    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 20
    lbfgs_max_eval: int | None = None
    lbfgs_history_size: int = 100
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    lbfgs_line_search_fn: str | None = "strong_wolfe"
    log_every: int = 100
    relative_l2_every: int = 1
    relative_l2_chunk_size: int = 65536
    agent_update_interval: int = 25
    agent_warmup_steps: int = 0
    compile_model: bool = False


@dataclass
class ExperimentConfig:
    equation: str = "burgers"
    equation_params: dict[str, Any] = field(default_factory=dict)
    controllers: list[str] = field(
        default_factory=lambda: [
            "fixed",
            "tiny_loss_weight",
            "softadapt",
            "relobralo",
            "gradnorm",
        ],
    )
    seed: int = 1234
    device: str = "auto"
    output_dir: str = "artifacts"
    save_plots: bool = True
    plot_grid: int = 120
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    controller_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        data = dict(payload)
        model = ModelConfig(**data.pop("model", {}))
        training = TrainingConfig(**data.pop("training", {}))
        return cls(model=model, training=training, **data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_cli_overrides(
        self,
        *,
        equation: str | None = None,
        controllers: str | None = None,
        steps: int | None = None,
        device: str | None = None,
        seed: int | None = None,
        output_dir: str | None = None,
        reward: str | None = None,
        optimizer_mode: str | None = None,
        adam_steps: int | None = None,
        lbfgs_steps: int | None = None,
        compile_model: bool = False,
        save_plots: bool | None = None,
    ) -> "ExperimentConfig":
        data = self.to_dict()
        if equation is not None:
            data["equation"] = equation
        if controllers is not None:
            data["controllers"] = [
                item.strip() for item in controllers.split(",") if item.strip()
            ]
        if device is not None:
            data["device"] = device
        if seed is not None:
            data["seed"] = seed
        if output_dir is not None:
            data["output_dir"] = output_dir
        if save_plots is not None:
            data["save_plots"] = save_plots
        if steps is not None:
            data["training"]["steps"] = steps
        if optimizer_mode is not None:
            data["training"]["optimizer_mode"] = optimizer_mode
        if adam_steps is not None:
            data["training"]["adam_steps"] = adam_steps
        if lbfgs_steps is not None:
            data["training"]["lbfgs_steps"] = lbfgs_steps
        if compile_model:
            data["training"]["compile_model"] = True
        if reward is not None:
            params = dict(data.get("controller_params", {}))
            for controller in data["controllers"]:
                if controller.lower() in AGENT_CONTROLLER_NAMES:
                    controller_cfg = dict(params.get(controller, {}))
                    controller_cfg["reward"] = reward
                    params[controller] = controller_cfg
            data["controller_params"] = params
        return ExperimentConfig.from_dict(data)
