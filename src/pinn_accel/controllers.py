from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .agents import AGENT_NAMES, BaseWeightAgent, make_agent
from .rewards import Reward, RewardContext, make_reward


def normalize_weights(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 1e-8, None)
    return (clipped / float(np.sum(clipped))).astype(np.float32)


def last_linear_weight(model: nn.Module) -> torch.nn.Parameter:
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return module.weight
    params = [param for param in model.parameters() if param.requires_grad]
    if not params:
        raise RuntimeError("Model has no trainable parameters")
    return params[-1]


@dataclass(frozen=True)
class StepSnapshot:
    step: int
    total: float
    losses: np.ndarray
    weights: np.ndarray
    relative_l2: float | None
    progress: float
    agent_progress: float
    done: bool


class WeightController(nn.Module):
    name = "base"
    trainable = False
    uses_agent = False

    def __init__(self):
        super().__init__()
        self.component_names: list[str] = []
        self.register_buffer("base_weights", torch.empty(0, dtype=torch.float32))
        self.register_buffer("effective_weights", torch.empty(0, dtype=torch.float32))

    def bind(
        self,
        component_names: list[str],
        initial_weights: np.ndarray,
        device: torch.device,
    ) -> None:
        self.component_names = list(component_names)
        weights = torch.tensor(
            normalize_weights(initial_weights),
            dtype=torch.float32,
            device=device,
        )
        self.base_weights = weights
        self.effective_weights = weights.clone()
        self._bind_state(device)

    def _bind_state(self, device: torch.device) -> None:
        del device

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def frozen_objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model, step
        weights = self.effective_weights.to(losses.device)
        return torch.sum(weights.detach() * losses), weights.detach()

    def after_step(
        self,
        snapshot: StepSnapshot,
        baseline_history: dict[str, Any] | None,
    ) -> dict[str, float | None]:
        del snapshot, baseline_history
        return {}

    def frozen_step_extras(self) -> dict[str, float | None]:
        return {}


class FixedController(WeightController):
    name = "fixed"

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model, step, update_state
        weights = self.base_weights.to(losses.device)
        self.effective_weights.copy_(weights.detach())
        return torch.sum(weights * losses), weights.detach()

    def frozen_objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.objective(losses, model, step, update_state=False)


class SoftAdaptController(WeightController):
    name = "softadapt"

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)

    def _bind_state(self, device: torch.device) -> None:
        self.register_buffer(
            "previous_losses",
            torch.zeros(len(self.component_names), dtype=torch.float32, device=device),
        )

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        if not update_state:
            weights = self.effective_weights.to(losses.device)
            return torch.sum(weights.detach() * losses), weights.detach()
        if step == 1:
            self.previous_losses.copy_(losses.detach())
            weights = self.base_weights.to(losses.device)
        else:
            with torch.no_grad():
                ratios = losses.detach() / torch.clamp(self.previous_losses, min=1e-8)
                scores = ratios / self.temperature
                lam = torch.exp(scores - torch.max(scores))
                lam = len(lam) * lam / torch.clamp(lam.sum(), min=1e-8)
                self.previous_losses.copy_(losses.detach())
            weights = self.base_weights.to(losses.device) * lam
        self.effective_weights.copy_(weights.detach())
        return torch.sum(weights.detach() * losses), weights.detach()

    def frozen_objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model, step
        weights = self.effective_weights.to(losses.device)
        return torch.sum(weights.detach() * losses), weights.detach()


class ReLoBRaLoController(WeightController):
    name = "relobralo"

    def __init__(self, alpha: float = 0.999, beta: float = 0.99, tau: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)

    def _bind_state(self, device: torch.device) -> None:
        n = len(self.component_names)
        self.register_buffer(
            "initial_losses",
            torch.zeros(n, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "previous_losses",
            torch.zeros(n, dtype=torch.float32, device=device),
        )
        self.register_buffer("lambda_ema", torch.ones(n, dtype=torch.float32, device=device))

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model
        if not update_state:
            weights = self.effective_weights.to(losses.device)
            return torch.sum(weights.detach() * losses), weights.detach()
        if step == 1:
            clamped = torch.clamp(losses.detach(), min=1e-8)
            self.initial_losses.copy_(clamped)
            self.previous_losses.copy_(clamped)
            weights = self.base_weights.to(losses.device)
        else:
            with torch.no_grad():
                ratios_prev = losses.detach() / (self.tau * self.previous_losses + 1e-8)
                ratios_init = losses.detach() / (self.tau * self.initial_losses + 1e-8)
                lam_prev = torch.exp(ratios_prev - torch.max(ratios_prev))
                lam_init = torch.exp(ratios_init - torch.max(ratios_init))
                lam_prev = len(lam_prev) * lam_prev / torch.clamp(lam_prev.sum(), min=1e-8)
                lam_init = len(lam_init) * lam_init / torch.clamp(lam_init.sum(), min=1e-8)
                rho = torch.bernoulli(torch.tensor(self.beta, device=losses.device))
                self.lambda_ema.copy_(
                    self.alpha * (rho * self.lambda_ema + (1.0 - rho) * lam_init)
                    + (1.0 - self.alpha) * lam_prev
                )
                self.previous_losses.copy_(torch.clamp(losses.detach(), min=1e-8))
            weights = self.base_weights.to(losses.device) * self.lambda_ema
        self.effective_weights.copy_(weights.detach())
        return torch.sum(weights.detach() * losses), weights.detach()


class GradNormController(WeightController):
    name = "gradnorm"
    trainable = True

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self.log_lambda: nn.Parameter | None = None

    def _bind_state(self, device: torch.device) -> None:
        self.register_buffer(
            "initial_losses",
            torch.zeros(len(self.component_names), dtype=torch.float32, device=device),
        )
        self.log_lambda = nn.Parameter(torch.zeros(len(self.component_names), device=device))

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.log_lambda is None:
            raise RuntimeError("GradNormController is not bound")
        if step == 1 and update_state:
            self.initial_losses.copy_(torch.clamp(losses.detach(), min=1e-8))

        if update_state:
            with torch.no_grad():
                normalizer = len(losses) / torch.clamp(
                    torch.exp(self.log_lambda).sum(),
                    min=1e-8,
                )
                self.log_lambda.add_(torch.log(normalizer))

        lam = torch.exp(self.log_lambda)
        relative = losses.detach() / torch.clamp(self.initial_losses, min=1e-8)
        inverse_rate = relative / torch.clamp(relative.mean(), min=1e-8)
        target_grad = torch.pow(inverse_rate, self.alpha)
        shared = last_linear_weight(model)
        grad_norms = torch.zeros_like(losses)
        for idx in range(len(losses)):
            grad = torch.autograd.grad(losses[idx], shared, retain_graph=True)[0]
            grad_norms[idx] = torch.norm(lam[idx] * grad.detach(), p=2)
        avg_grad = grad_norms.detach().mean()
        grad_loss = torch.abs(grad_norms - avg_grad * target_grad).sum()
        weights = self.base_weights.to(losses.device) * lam
        model_loss = torch.sum(weights.detach() * losses)
        self.effective_weights.copy_(weights.detach())
        return model_loss + grad_loss, weights.detach()

    def frozen_objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model, step
        if self.log_lambda is None:
            raise RuntimeError("GradNormController is not bound")
        with torch.no_grad():
            lam = torch.exp(self.log_lambda.detach())
            lam = len(lam) * lam / torch.clamp(lam.sum(), min=1e-8)
            weights = self.base_weights.to(losses.device) * lam.to(losses.device)
            self.effective_weights.copy_(weights.detach())
        return torch.sum(weights.detach() * losses), weights.detach()


class AgentWeightController(WeightController):
    name = "agent"
    uses_agent = True

    def __init__(
        self,
        agent: BaseWeightAgent,
        reward: Reward,
        update_interval: int,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.agent = agent
        self.reward = reward
        self.update_interval = int(update_interval)
        self.warmup_steps = int(warmup_steps)
        self._weights_np: np.ndarray | None = None
        self._initial_snapshot: StepSnapshot | None = None
        self._previous_state: np.ndarray | None = None
        self._previous_action: np.ndarray | None = None
        self._previous_snapshot: StepSnapshot | None = None
        self.name = f"{type(agent).__name__.replace('Agent', '').lower()}_{reward.name}"

    @property
    def requires_baseline(self) -> bool:
        return self.reward.requires_baseline

    def bind(
        self,
        component_names: list[str],
        initial_weights: np.ndarray,
        device: torch.device,
    ) -> None:
        super().bind(component_names, initial_weights, device)
        self._weights_np = normalize_weights(initial_weights)
        self.agent.bind(component_names, device)
        self.agent.set_weight_reference(self._weights_np)

    def objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
        update_state: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del model, step, update_state
        if self._weights_np is None:
            raise RuntimeError("AgentWeightController is not bound")
        weights = torch.tensor(self._weights_np, dtype=losses.dtype, device=losses.device)
        self.effective_weights.copy_(weights.detach())
        return torch.sum(weights * losses), weights.detach()

    def frozen_objective(
        self,
        losses: torch.Tensor,
        model: nn.Module,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.objective(losses, model, step, update_state=False)

    def after_step(
        self,
        snapshot: StepSnapshot,
        baseline_history: dict[str, Any] | None,
    ) -> dict[str, float | None]:
        if self._weights_np is None:
            raise RuntimeError("AgentWeightController is not bound")
        sigma = self._agent_sigma()
        if self._initial_snapshot is None:
            self._initial_snapshot = snapshot
        if snapshot.step < self.warmup_steps:
            return {"agent_reward": None, "agent_sigma": sigma}
        if snapshot.step % self.update_interval != 0 and not snapshot.done:
            return {"agent_reward": None, "agent_sigma": sigma}

        state = self.agent.make_state(
            snapshot.losses,
            self._weights_np,
            snapshot.agent_progress,
        )
        reward_value = None
        if (
            self._previous_state is not None
            and self._previous_action is not None
            and self._previous_snapshot is not None
        ):
            ctx = self._make_reward_context(snapshot, baseline_history)
            reward_value = self.reward(ctx)
            self.agent.update(
                self._previous_state,
                self._previous_action,
                reward_value,
                state,
                snapshot.done,
            )

        action = self.agent.select_action(state)
        self._weights_np = self.agent.apply_action(self._weights_np, action)
        self.agent.prev_losses = snapshot.losses.copy()
        self._previous_state = state
        self._previous_action = np.asarray(action, dtype=np.float32).copy()
        self._previous_snapshot = snapshot
        return {"agent_reward": reward_value, "agent_sigma": self._agent_sigma()}

    def frozen_step_extras(self) -> dict[str, float | None]:
        return {"agent_reward": None, "agent_sigma": self._agent_sigma()}

    def _agent_sigma(self) -> float | None:
        sigma_getter = getattr(self.agent, "current_sigma", None)
        if sigma_getter is None:
            return None
        return sigma_getter()

    def _make_reward_context(
        self,
        snapshot: StepSnapshot,
        baseline_history: dict[str, Any] | None,
    ) -> RewardContext:
        if self._previous_snapshot is None:
            raise RuntimeError("previous snapshot is missing")
        if self._initial_snapshot is None:
            raise RuntimeError("initial snapshot is missing")
        initial = self._initial_snapshot
        previous = self._previous_snapshot
        baseline_initial = None
        baseline_previous = None
        baseline_current = None
        baseline_initial_relative_l2 = None
        baseline_previous_relative_l2 = None
        baseline_current_relative_l2 = None
        baseline_previous_losses = None
        baseline_current_losses = None
        if self.reward.requires_baseline:
            if baseline_history is None:
                raise ValueError(f"{self.reward.name} requires a baseline history")
            baseline_initial = _history_total_at(baseline_history, initial.step)
            baseline_previous = _history_total_at(baseline_history, previous.step)
            baseline_current = _history_total_at(baseline_history, snapshot.step)
            baseline_initial_relative_l2 = _history_relative_l2_at(
                baseline_history,
                initial.step,
            )
            baseline_previous_relative_l2 = _history_relative_l2_at(
                baseline_history,
                previous.step,
            )
            baseline_current_relative_l2 = _history_relative_l2_at(
                baseline_history,
                snapshot.step,
            )
            baseline_previous_losses = _history_losses_at(
                baseline_history,
                previous.step,
                self.component_names,
            )
            baseline_current_losses = _history_losses_at(
                baseline_history,
                snapshot.step,
                self.component_names,
            )
        return RewardContext(
            initial_total=initial.total,
            previous_total=previous.total,
            current_total=snapshot.total,
            previous_losses=previous.losses,
            current_losses=snapshot.losses,
            progress=snapshot.agent_progress,
            initial_relative_l2=initial.relative_l2,
            previous_relative_l2=previous.relative_l2,
            current_relative_l2=snapshot.relative_l2,
            baseline_initial_total=baseline_initial,
            baseline_previous_total=baseline_previous,
            baseline_current_total=baseline_current,
            baseline_initial_relative_l2=baseline_initial_relative_l2,
            baseline_previous_relative_l2=baseline_previous_relative_l2,
            baseline_current_relative_l2=baseline_current_relative_l2,
            baseline_previous_losses=baseline_previous_losses,
            baseline_current_losses=baseline_current_losses,
        )


def _history_total_at(history: dict[str, Any], step: int) -> float:
    values = history["equal_weight_total"]
    index = min(max(step - 1, 0), len(values) - 1)
    return float(values[index])


def _history_losses_at(history: dict[str, Any], step: int, names: list[str]) -> np.ndarray:
    index = min(max(step - 1, 0), len(history["equal_weight_total"]) - 1)
    return np.array([history["components"][name][index] for name in names], dtype=np.float64)


def _history_relative_l2_at(history: dict[str, Any], step: int) -> float | None:
    values = history.get("relative_l2", [])
    if not values:
        return None
    index = min(max(step - 1, 0), len(values) - 1)
    value = values[index]
    return None if value is None else float(value)


def make_controller(
    name: str,
    params: dict[str, Any],
    *,
    update_interval: int,
    warmup_steps: int,
) -> WeightController:
    value = name.lower()
    cfg = dict(params)
    if value == "fixed":
        return FixedController()
    if value == "softadapt":
        return SoftAdaptController(**cfg)
    if value == "relobralo":
        return ReLoBRaLoController(**cfg)
    if value == "gradnorm":
        return GradNormController(**cfg)
    if value in AGENT_NAMES:
        reward_name = str(cfg.pop("reward", "log_ratio"))
        reward_params = dict(cfg.pop("reward_params", {}))
        reward = make_reward(reward_name, reward_params)
        agent = make_agent(value, cfg)
        return AgentWeightController(
            agent,
            reward,
            update_interval=update_interval,
            warmup_steps=warmup_steps,
        )
    raise ValueError(f"Unknown controller: {name}")


def controller_needs_baseline(name: str, params: dict[str, Any]) -> bool:
    if name.lower() not in AGENT_NAMES:
        return False
    reward_name = str(params.get("reward", "log_ratio"))
    reward = make_reward(reward_name, params.get("reward_params", {}))
    return reward.requires_baseline
