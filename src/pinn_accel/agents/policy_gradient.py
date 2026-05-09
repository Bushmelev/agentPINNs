from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..models import MLP
from .base import BaseWeightAgent


def _atanh(x: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(x, -0.999999, 0.999999)
    return 0.5 * (torch.log1p(clipped) - torch.log1p(-clipped))


def squashed_log_prob(dist: torch.distributions.Normal, action: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(action, -0.999999, 0.999999)
    pre_tanh = _atanh(clipped)
    correction = torch.log(torch.clamp(1.0 - clipped.pow(2), min=1e-6))
    return (dist.log_prob(pre_tanh) - correction).sum(dim=1)


class PolicyGradientAgent(BaseWeightAgent):
    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (32, 32),
        activation: str = "tanh",
        init_std: float = 0.05,
        baseline_beta: float = 0.9,
        entropy_coef: float = 0.0,
        zero_init_policy: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.init_std = float(init_std)
        self.baseline_beta = float(baseline_beta)
        self.entropy_coef = float(entropy_coef)
        self.zero_init_policy = bool(zero_init_policy)
        self.policy: MLP | None = None
        self.log_std: nn.Parameter | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.baseline = 0.0

    def _build_networks(self) -> None:
        self.policy = MLP(
            [self.state_dim(), *self.hidden_sizes, self.action_dim],
            activation=self.activation,
        ).to(self.device)
        if self.zero_init_policy:
            output_layer = self.policy.net[-1]
            if isinstance(output_layer, nn.Linear):
                nn.init.zeros_(output_layer.weight)
                nn.init.zeros_(output_layer.bias)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), np.log(self.init_std), device=self.device)
        )
        self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        if self.policy is None or self.log_std is None:
            return
        self.optimizer = self._make_optimizer([*self.policy.parameters(), self.log_std])

    def _distribution(self, state_t: torch.Tensor) -> torch.distributions.Normal:
        if self.policy is None or self.log_std is None:
            raise RuntimeError("PolicyGradientAgent is not bound")
        mean = self.policy(state_t)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist = self._distribution(state_t)
            pre_tanh = dist.sample() if self.trainable else dist.mean
            action = torch.tanh(pre_tanh)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        del next_state, done
        if not self.trainable or self.optimizer is None:
            return
        self.baseline = self.baseline_beta * self.baseline + (1.0 - self.baseline_beta) * reward
        advantage = reward - self.baseline
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self._distribution(state_t)
        loss = -(squashed_log_prob(dist, action_t) * advantage).mean()
        if self.entropy_coef:
            loss -= self.entropy_coef * dist.entropy().sum(dim=1).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
