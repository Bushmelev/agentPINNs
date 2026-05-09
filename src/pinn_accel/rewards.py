from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    previous_total: float
    current_total: float
    previous_losses: np.ndarray
    current_losses: np.ndarray
    baseline_previous_total: float | None = None
    baseline_current_total: float | None = None
    baseline_previous_losses: np.ndarray | None = None
    baseline_current_losses: np.ndarray | None = None


class Reward:
    name = "reward"
    requires_baseline = False

    def __call__(self, ctx: RewardContext) -> float:
        raise NotImplementedError


class LogRatioReward(Reward):
    name = "log_ratio"

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        return float(np.clip(np.log((ctx.previous_total + eps) / (ctx.current_total + eps)), -1.0, 1.0))


class RelativeImprovementReward(Reward):
    name = "relative_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        return float((ctx.previous_total - ctx.current_total) / (abs(ctx.previous_total) + eps))


class ComponentRelativeImprovementReward(Reward):
    name = "component_relative_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        values = (ctx.previous_losses - ctx.current_losses) / (np.abs(ctx.previous_losses) + eps)
        return float(np.mean(values))


class BaselineGapReward(Reward):
    name = "baseline_gap"
    requires_baseline = True

    def __init__(self, alpha: float = 0.25):
        self.alpha = float(alpha)

    def __call__(self, ctx: RewardContext) -> float:
        if ctx.baseline_previous_total is None or ctx.baseline_current_total is None:
            raise ValueError("baseline_gap reward requires baseline totals")
        eps = 1e-8
        agent_progress = np.log((ctx.previous_total + eps) / (ctx.current_total + eps))
        baseline_progress = np.log(
            (ctx.baseline_previous_total + eps) / (ctx.baseline_current_total + eps)
        )
        progress_delta = float(agent_progress - baseline_progress)
        previous_gap = np.log((ctx.baseline_previous_total + eps) / (ctx.previous_total + eps))
        current_gap = np.log((ctx.baseline_current_total + eps) / (ctx.current_total + eps))
        gap_delta = float(current_gap - previous_gap)
        return float(self.alpha * progress_delta + (1.0 - self.alpha) * gap_delta)


REWARD_REGISTRY = {
    "log_ratio": LogRatioReward,
    "relative_improvement": RelativeImprovementReward,
    "component_relative_improvement": ComponentRelativeImprovementReward,
    "baseline_gap": BaselineGapReward,
}


def make_reward(name: str, params: dict | None = None) -> Reward:
    key = name.lower()
    if key not in REWARD_REGISTRY:
        known = ", ".join(sorted(REWARD_REGISTRY))
        raise ValueError(f"Unknown reward {name!r}. Known rewards: {known}")
    kwargs = dict(params or {})
    return REWARD_REGISTRY[key](**kwargs)
