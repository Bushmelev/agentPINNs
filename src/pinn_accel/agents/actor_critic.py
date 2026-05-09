from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..models import get_activation
from .base import BaseWeightAgent
from .policy_gradient import squashed_log_prob


class ActorCriticNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: tuple[int, ...],
        activation: str,
        action_dim: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(previous, hidden))
            layers.append(get_activation(activation))
            previous = hidden
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(previous, action_dim)
        self.value_head = nn.Linear(previous, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(state)
        return self.policy_head(features), self.value_head(features)


class ActorCriticAgent(BaseWeightAgent):
    def __init__(
        self,
        *,
        hidden_sizes: tuple[int, ...] = (64, 64),
        activation: str = "tanh",
        gamma: float = 0.95,
        init_std: float = 0.05,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        zero_init_policy: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation
        self.gamma = float(gamma)
        self.init_std = float(init_std)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)
        self.zero_init_policy = bool(zero_init_policy)
        self.net: ActorCriticNet | None = None
        self.log_std: nn.Parameter | None = None
        self.optimizer: torch.optim.Optimizer | None = None

    def _build_networks(self) -> None:
        self.net = ActorCriticNet(
            self.state_dim(),
            self.hidden_sizes,
            self.activation,
            self.action_dim,
        ).to(self.device)
        if self.zero_init_policy:
            nn.init.zeros_(self.net.policy_head.weight)
            nn.init.zeros_(self.net.policy_head.bias)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), np.log(self.init_std), device=self.device)
        )
        self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        if self.net is None or self.log_std is None:
            return
        self.optimizer = self._make_optimizer([*self.net.parameters(), self.log_std])

    def _distribution(
        self,
        state_t: torch.Tensor,
    ) -> tuple[torch.distributions.Normal, torch.Tensor]:
        if self.net is None or self.log_std is None:
            raise RuntimeError("ActorCriticAgent is not bound")
        mean, value = self.net(state_t)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).unsqueeze(0).expand_as(mean)
        return torch.distributions.Normal(mean, std), value

    def select_action(self, state: np.ndarray) -> np.ndarray:
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            dist, _ = self._distribution(state_t)
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
        if not self.trainable or self.optimizer is None or self.net is None:
            return
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.tensor(
            next_state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        action_t = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist, value = self._distribution(state_t)
        with torch.no_grad():
            _, next_value = self.net(next_state_t)
            not_done = 0.0 if done else 1.0
            target = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
            target = target + self.gamma * not_done * next_value
        advantage = target - value
        policy_loss = -(squashed_log_prob(dist, action_t) * advantage.detach().squeeze(-1)).mean()
        value_loss = torch.mean((value - target.detach()) ** 2)
        entropy = dist.entropy().sum(dim=1).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
