from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .base import BaseWeightAgent


class LinearRLPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        sigma: float = 0.1,
        bias: bool = False,
        hidden_dim: int | None = None,
        learn_sigma: bool = False,
        sigma_min: float = 1e-4,
        sigma_max: float | None = None,
    ):
        super().__init__()
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        if sigma_min <= 0.0:
            raise ValueError("sigma_min must be positive")
        if sigma_max is not None and sigma_max <= sigma_min:
            raise ValueError("sigma_max must be greater than sigma_min")
        if hidden_dim is not None and hidden_dim <= 0:
            hidden_dim = None
        self.hidden_dim = hidden_dim
        self.learn_sigma = bool(learn_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        if hidden_dim is None:
            self.net = nn.Linear(state_dim, action_dim, bias=bias)
        else:
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim, bias=bias),
                nn.Tanh(),
                nn.Linear(hidden_dim, action_dim, bias=bias),
            )
        initial_log_sigma = math.log(float(sigma))
        if self.learn_sigma:
            self.log_sigma = nn.Parameter(
                torch.tensor(initial_log_sigma, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "log_sigma",
                torch.tensor(initial_log_sigma, dtype=torch.float32),
            )

    def mean(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state))

    def sigma_value(self) -> torch.Tensor:
        sigma = torch.exp(self.log_sigma)
        if self.sigma_max is None:
            return torch.clamp(sigma, min=self.sigma_min)
        return torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max)

    def distribution(self, state: torch.Tensor) -> torch.distributions.Normal:
        mu = self.mean(state)
        std = self.sigma_value().to(dtype=mu.dtype, device=mu.device).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(state)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def log_prob(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).log_prob(action).sum(dim=-1)

    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        return self.distribution(state).entropy().sum(dim=-1)


class TinyLossWeightAgent(BaseWeightAgent):
    def __init__(
        self,
        *,
        sigma: float = 0.1,
        baseline_beta: float = 0.9,
        entropy_coef: float = 0.0,
        zero_init_policy: bool = True,
        policy_bias: bool = False,
        policy_hidden_dim: int | None = None,
        learn_sigma: bool = False,
        sigma_min: float = 1e-4,
        sigma_max: float | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sigma = float(sigma)
        self.learn_sigma = bool(learn_sigma)
        self.sigma_min = float(sigma_min)
        self.sigma_max = None if sigma_max is None else float(sigma_max)
        self.baseline_beta = float(baseline_beta)
        self.entropy_coef = float(entropy_coef)
        self.zero_init_policy = bool(zero_init_policy)
        self.policy_bias = bool(policy_bias)
        self.policy_hidden_dim = policy_hidden_dim
        self.policy: LinearRLPolicy | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.baseline = 0.0

    def _build_networks(self) -> None:
        self.policy = LinearRLPolicy(
            self.state_dim(),
            self.action_dim,
            self.sigma,
            bias=self.policy_bias,
            hidden_dim=self.policy_hidden_dim,
            learn_sigma=self.learn_sigma,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
        ).to(self.device)
        if self.zero_init_policy:
            output_layer = self._output_layer()
            nn.init.zeros_(output_layer.weight)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)
        self._rebuild_optimizer()

    def _output_layer(self) -> nn.Linear:
        if self.policy is None:
            raise RuntimeError("TinyLossWeightAgent is not bound")
        if isinstance(self.policy.net, nn.Linear):
            return self.policy.net
        return self.policy.net[-1]

    def _rebuild_optimizer(self) -> None:
        if self.policy is None:
            return
        self.optimizer = self._make_optimizer(self.policy.parameters())

    def forward(
        self,
        log_losses: torch.Tensor,
        dlog_losses: torch.Tensor,
        log_lambdas: torch.Tensor,
        progress: torch.Tensor,
        log_initial_loss_ratios: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.policy is None:
            raise RuntimeError("TinyLossWeightAgent is not bound")
        progress_t = torch.as_tensor(
            progress,
            dtype=log_losses.dtype,
            device=log_losses.device,
        ).reshape(1)
        pieces = []
        if self.include_log_losses:
            pieces.append(log_losses)
        pieces.extend([dlog_losses, log_lambdas])
        if self.include_initial_loss_ratios:
            if log_initial_loss_ratios is None:
                log_initial_loss_ratios = torch.zeros_like(log_losses)
            pieces.append(log_initial_loss_ratios)
        state = torch.cat([*pieces, progress_t])
        if self.trainable:
            action, _ = self.policy(state)
        else:
            action = self.policy.mean(state)
        return torch.clamp(action, -1.0, 1.0)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("TinyLossWeightAgent is not bound")
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            if self.trainable:
                action, _ = self.policy(state_t)
            else:
                action = self.policy.mean(state_t)
        return action.cpu().numpy().astype(np.float32)

    def current_sigma(self) -> float | None:
        if self.policy is None:
            return None
        return float(self.policy.sigma_value().detach().cpu().item())

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        del next_state, done
        if not self.trainable or self.optimizer is None or self.policy is None:
            return
        self.baseline = self.baseline_beta * self.baseline + (1.0 - self.baseline_beta) * reward
        advantage = float(reward - self.baseline)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device)
        loss = -self.policy.log_prob(state_t, action_t) * advantage
        if self.entropy_coef:
            loss -= self.entropy_coef * self.policy.entropy(state_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
