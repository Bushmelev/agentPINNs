from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..optim import make_optimizer


class BaseWeightAgent(nn.Module):
    def __init__(
        self,
        *,
        action_scale: float = 0.2,
        min_weight: float = 1e-6,
        min_weight_share: float | None = None,
        max_weight_share: float | None = None,
        include_initial_loss_ratios: bool = True,
        feature_clip: float = 10.0,
        trainable: bool = True,
        optimizer: str = "adam",
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.action_scale = float(action_scale)
        self.min_weight = float(min_weight)
        self.min_weight_share = min_weight_share
        self.max_weight_share = max_weight_share
        self.include_initial_loss_ratios = bool(include_initial_loss_ratios)
        self.feature_clip = float(feature_clip)
        self.trainable = trainable
        self.optimizer_name = optimizer
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.component_names: list[str] = []
        self.action_dim = 0
        self.target_weight_sum: float | None = None
        self.device = torch.device("cpu")
        self.prev_losses: np.ndarray | None = None
        self.initial_losses: np.ndarray | None = None

    def bind(self, component_names: list[str], device: torch.device) -> None:
        if self.component_names:
            if self.component_names != component_names:
                raise ValueError(
                    f"Agent already bound to {self.component_names}, got {component_names}"
                )
            return
        self.component_names = list(component_names)
        self.action_dim = len(component_names)
        self.device = device
        self._build_networks()

    def _build_networks(self) -> None:
        return None

    def _make_optimizer(self, params) -> torch.optim.Optimizer:
        return make_optimizer(
            params,
            self.optimizer_name,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def state_dim(self) -> int:
        n_losses = len(self.component_names)
        ratio_features = n_losses if self.include_initial_loss_ratios else 0
        return 3 * n_losses + ratio_features + 1

    def configure_optimizer(self, **kwargs: Any) -> None:
        if "optimizer" in kwargs:
            self.optimizer_name = str(kwargs["optimizer"])
        if "lr" in kwargs:
            self.lr = float(kwargs["lr"])
        if "weight_decay" in kwargs:
            self.weight_decay = float(kwargs["weight_decay"])
        self._rebuild_optimizer()

    def _rebuild_optimizer(self) -> None:
        return None

    def set_weight_reference(self, weights: np.ndarray) -> None:
        clipped = np.clip(np.asarray(weights, dtype=np.float32), self.min_weight, None)
        self.target_weight_sum = float(np.sum(clipped))

    def make_state(
        self,
        losses: np.ndarray,
        weights: np.ndarray,
        progress: float,
    ) -> np.ndarray:
        eps = 1e-8
        losses_np = np.clip(np.asarray(losses, dtype=np.float32), eps, None)
        weights_np = np.asarray(weights, dtype=np.float32)
        if self.initial_losses is None:
            self.initial_losses = losses_np.copy()

        mean_loss = float(np.mean(losses_np))
        log_losses = np.log((losses_np + eps) / (mean_loss + eps)).astype(np.float32)
        if self.prev_losses is None:
            dlog_losses = np.zeros_like(losses_np, dtype=np.float32)
        else:
            previous = np.clip(self.prev_losses, eps, None)
            dlog_losses = np.log((previous + eps) / (losses_np + eps)).astype(np.float32)
        weight_sum = max(float(np.sum(weights_np)), eps)
        normalized_weights = (weights_np / weight_sum).astype(np.float32)
        mean_weight = 1.0 / max(len(normalized_weights), 1)
        log_lambdas = np.log(
            (np.clip(normalized_weights, eps, None) + eps) / (mean_weight + eps)
        ).astype(np.float32)

        pieces = [log_losses, dlog_losses, log_lambdas]
        if self.include_initial_loss_ratios:
            initial = np.clip(self.initial_losses, eps, None)
            pieces.append(np.log((losses_np + eps) / (initial + eps)).astype(np.float32))
        pieces.append(np.array([np.clip(progress, 0.0, 1.0)], dtype=np.float32))
        state = np.concatenate(pieces).astype(np.float32)
        if self.feature_clip > 0.0:
            state = np.clip(state, -self.feature_clip, self.feature_clip).astype(np.float32)
        return state

    def split_state(self, state: np.ndarray) -> tuple[np.ndarray, ...]:
        n_losses = len(self.component_names)
        parts: list[np.ndarray] = [
            state[0:n_losses],
            state[n_losses : 2 * n_losses],
            state[2 * n_losses : 3 * n_losses],
        ]
        cursor = 3 * n_losses
        if self.include_initial_loss_ratios:
            parts.append(state[cursor : cursor + n_losses])
            cursor += n_losses
        parts.append(state[cursor : cursor + 1])
        return tuple(parts)

    def apply_action(self, weights: np.ndarray, action: np.ndarray) -> np.ndarray:
        action_np = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        updated = np.clip(weights, self.min_weight, None) * np.exp(
            self.action_scale * action_np
        )
        return self._project_weights(updated.astype(np.float32))

    def _project_weights(self, weights: np.ndarray) -> np.ndarray:
        if self.target_weight_sum is None:
            return np.clip(weights, self.min_weight, None).astype(np.float32)

        target = float(self.target_weight_sum)
        lower = max(self.min_weight, target * float(self.min_weight_share or 0.0))
        upper = (
            float("inf")
            if self.max_weight_share is None
            else target * self.max_weight_share
        )
        projected = np.clip(weights, lower, upper).astype(np.float32)

        for _ in range(32):
            delta = target - float(np.sum(projected))
            if abs(delta) <= 1e-6:
                return projected.astype(np.float32)
            if delta > 0:
                slack = upper - projected
                eligible = (
                    np.ones_like(projected, dtype=bool)
                    if np.isinf(upper)
                    else slack > 1e-8
                )
                allocation = (
                    np.maximum(projected[eligible], 1.0)
                    if np.isinf(upper)
                    else slack[eligible]
                )
                projected[eligible] += delta * allocation / float(np.sum(allocation))
                projected = np.minimum(projected, upper)
            else:
                removable = projected - lower
                eligible = removable > 1e-8
                projected[eligible] += (
                    delta * removable[eligible] / float(np.sum(removable[eligible]))
                )
                projected = np.maximum(projected, lower)
        raise RuntimeError("Failed to project agent weights")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def update(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        raise NotImplementedError
