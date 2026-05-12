from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    initial_total: float
    previous_total: float
    current_total: float
    initial_losses: np.ndarray
    previous_losses: np.ndarray
    current_losses: np.ndarray
    progress: float = 0.0
    initial_relative_l2: float | None = None
    previous_relative_l2: float | None = None
    current_relative_l2: float | None = None
    baseline_initial_total: float | None = None
    baseline_previous_total: float | None = None
    baseline_current_total: float | None = None
    baseline_initial_relative_l2: float | None = None
    baseline_previous_relative_l2: float | None = None
    baseline_current_relative_l2: float | None = None
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
        return float(
            np.clip(
                np.log((ctx.previous_total + eps) / (ctx.current_total + eps)),
                -1.0,
                1.0,
            )
        )


class RelativeImprovementReward(Reward):
    name = "relative_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        return float((ctx.previous_total - ctx.current_total) / (abs(ctx.previous_total) + eps))


class ComponentRelativeImprovementReward(Reward):
    name = "component_relative_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        values = (ctx.previous_losses - ctx.current_losses) / (
            np.abs(ctx.previous_losses) + eps
        )
        return float(np.mean(values))


def _require_baseline_totals(ctx: RewardContext) -> tuple[float, float]:
    if ctx.baseline_initial_total is None or ctx.baseline_current_total is None:
        raise ValueError("baseline reward requires baseline totals")
    return ctx.baseline_initial_total, ctx.baseline_current_total


def _require_baseline_loss_pair(ctx: RewardContext) -> tuple[np.ndarray, np.ndarray]:
    if ctx.baseline_previous_losses is None or ctx.baseline_current_losses is None:
        raise ValueError("component baseline reward requires baseline losses")
    return ctx.baseline_previous_losses, ctx.baseline_current_losses


def _require_relative_l2(ctx: RewardContext) -> tuple[float, float, float]:
    if (
        ctx.initial_relative_l2 is None
        or ctx.previous_relative_l2 is None
        or ctx.current_relative_l2 is None
    ):
        raise ValueError("relative_l2 reward requires relative_l2 metrics")
    return ctx.initial_relative_l2, ctx.previous_relative_l2, ctx.current_relative_l2


def _require_baseline_relative_l2(ctx: RewardContext) -> tuple[float, float, float]:
    if (
        ctx.baseline_initial_relative_l2 is None
        or ctx.baseline_previous_relative_l2 is None
        or ctx.baseline_current_relative_l2 is None
    ):
        raise ValueError("relative_l2 baseline reward requires baseline relative_l2 metrics")
    return (
        ctx.baseline_initial_relative_l2,
        ctx.baseline_previous_relative_l2,
        ctx.baseline_current_relative_l2,
    )


def _normalized_baseline_gap(ctx: RewardContext) -> float:
    baseline_initial, baseline_current = _require_baseline_totals(ctx)
    eps = 1e-8
    baseline_norm = baseline_current / (baseline_initial + eps)
    agent_norm = ctx.current_total / (ctx.initial_total + eps)
    return float(baseline_norm - agent_norm)


def _relative_l2_baseline_gap(ctx: RewardContext) -> float:
    initial, _, current = _require_relative_l2(ctx)
    baseline_initial, _, baseline_current = _require_baseline_relative_l2(ctx)
    eps = 1e-8
    baseline_norm = baseline_current / (baseline_initial + eps)
    agent_norm = current / (initial + eps)
    return float(baseline_norm - agent_norm)


def _component_baseline_progress_delta(ctx: RewardContext) -> np.ndarray:
    baseline_previous, baseline_current = _require_baseline_loss_pair(ctx)
    eps = 1e-8
    agent_progress = np.log((ctx.previous_losses + eps) / (ctx.current_losses + eps))
    baseline_progress = np.log((baseline_previous + eps) / (baseline_current + eps))
    return agent_progress - baseline_progress


def _log_loss_progress(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    eps = 1e-8
    return np.log((previous + eps) / (current + eps))


def _log_ratio(previous: float, current: float) -> float:
    eps = 1e-8
    return float(np.log((previous + eps) / (current + eps)))


class RelativeL2ImprovementReward(Reward):
    name = "relative_l2_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        _, previous, current = _require_relative_l2(ctx)
        eps = 1e-8
        return float((previous - current) / (abs(previous) + eps))


class RelativeL2LogImprovementReward(Reward):
    name = "relative_l2_log_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        _, previous, current = _require_relative_l2(ctx)
        return _log_ratio(previous, current)


class WorstComponentRelativeImprovementReward(Reward):
    name = "worst_component_relative_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        return float(np.min(_log_loss_progress(ctx.previous_losses, ctx.current_losses)))


class ComponentBalancePenaltyReward(Reward):
    name = "component_balance_penalty"

    def __init__(self, beta: float = 0.1):
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        self.beta = float(beta)

    def __call__(self, ctx: RewardContext) -> float:
        eps = 1e-8
        progress_reward = float(
            np.mean(_log_loss_progress(ctx.previous_losses, ctx.current_losses))
        )
        component_ratios = np.log(
            (ctx.current_losses + eps) / (ctx.initial_losses + eps)
        )
        return float(progress_reward - self.beta * float(np.std(component_ratios)))


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


class NormalizedBaselineGapReward(Reward):
    name = "normalized_baseline_gap"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        return _normalized_baseline_gap(ctx)


class LogNormalizedBaselineGapReward(Reward):
    name = "log_normalized_baseline_gap"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        baseline_initial, baseline_current = _require_baseline_totals(ctx)
        eps = 1e-8
        baseline_norm = baseline_current / (baseline_initial + eps)
        agent_norm = ctx.current_total / (ctx.initial_total + eps)
        return float(np.log(baseline_norm + eps) - np.log(agent_norm + eps))


class ComponentBaselineGapReward(Reward):
    name = "component_baseline_gap"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        return float(np.mean(_component_baseline_progress_delta(ctx)))


class WorstComponentBaselineGapReward(Reward):
    name = "worst_component_baseline_gap"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        return float(np.min(_component_baseline_progress_delta(ctx)))


class NormalizedBaselineGapDeltaReward(Reward):
    name = "normalized_baseline_gap_delta"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        if (
            ctx.baseline_initial_total is None
            or ctx.baseline_previous_total is None
            or ctx.baseline_current_total is None
        ):
            raise ValueError("normalized_baseline_gap_delta reward requires baseline totals")
        eps = 1e-8
        previous_gap = (
            ctx.baseline_previous_total / (ctx.baseline_initial_total + eps)
            - ctx.previous_total / (ctx.initial_total + eps)
        )
        current_gap = (
            ctx.baseline_current_total / (ctx.baseline_initial_total + eps)
            - ctx.current_total / (ctx.initial_total + eps)
        )
        return float(current_gap - previous_gap)


class RelativeL2BaselineGapReward(Reward):
    name = "relative_l2_baseline_gap"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        return _relative_l2_baseline_gap(ctx)


class RelativeL2BaselineGapDeltaReward(Reward):
    name = "relative_l2_baseline_gap_delta"
    requires_baseline = True

    def __call__(self, ctx: RewardContext) -> float:
        initial, previous, current = _require_relative_l2(ctx)
        baseline_initial, baseline_previous, baseline_current = _require_baseline_relative_l2(
            ctx
        )
        eps = 1e-8
        previous_gap = baseline_previous / (baseline_initial + eps) - previous / (
            initial + eps
        )
        current_gap = baseline_current / (baseline_initial + eps) - current / (
            initial + eps
        )
        return float(current_gap - previous_gap)


class LossL2HybridReward(Reward):
    name = "loss_l2_hybrid"
    requires_baseline = True

    def __init__(self, beta: float = 0.5):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = float(beta)

    def __call__(self, ctx: RewardContext) -> float:
        loss_reward = _normalized_baseline_gap(ctx)
        l2_reward = _relative_l2_baseline_gap(ctx)
        return float(self.beta * loss_reward + (1.0 - self.beta) * l2_reward)


class LossL2SelfHybridReward(Reward):
    name = "loss_l2_self_hybrid"

    def __init__(self, beta: float = 0.5):
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        self.beta = float(beta)

    def __call__(self, ctx: RewardContext) -> float:
        _, previous_l2, current_l2 = _require_relative_l2(ctx)
        loss_reward = _log_ratio(ctx.previous_total, ctx.current_total)
        l2_reward = _log_ratio(previous_l2, current_l2)
        return float(self.beta * loss_reward + (1.0 - self.beta) * l2_reward)


class RunningBestL2Reward(Reward):
    name = "running_best_l2_reward"

    def __init__(self):
        self.best_relative_l2: float | None = None

    def __call__(self, ctx: RewardContext) -> float:
        initial, previous, current = _require_relative_l2(ctx)
        if self.best_relative_l2 is None:
            self.best_relative_l2 = min(float(initial), float(previous))
        reward = _log_ratio(self.best_relative_l2, current)
        self.best_relative_l2 = min(self.best_relative_l2, float(current))
        return reward


class ProgressiveLossL2HybridReward(Reward):
    name = "progressive_loss_l2_hybrid"
    requires_baseline = True

    def __init__(self, min_loss_weight: float = 0.0, max_loss_weight: float = 1.0):
        if not 0.0 <= min_loss_weight <= 1.0:
            raise ValueError("min_loss_weight must be in [0, 1]")
        if not 0.0 <= max_loss_weight <= 1.0:
            raise ValueError("max_loss_weight must be in [0, 1]")
        if min_loss_weight > max_loss_weight:
            raise ValueError("min_loss_weight must be <= max_loss_weight")
        self.min_loss_weight = float(min_loss_weight)
        self.max_loss_weight = float(max_loss_weight)

    def __call__(self, ctx: RewardContext) -> float:
        progress = float(np.clip(ctx.progress, 0.0, 1.0))
        loss_weight = self.max_loss_weight - (
            self.max_loss_weight - self.min_loss_weight
        ) * progress
        loss_reward = _normalized_baseline_gap(ctx)
        l2_reward = _relative_l2_baseline_gap(ctx)
        return float(loss_weight * loss_reward + (1.0 - loss_weight) * l2_reward)


REWARD_REGISTRY = {
    "log_ratio": LogRatioReward,
    "relative_improvement": RelativeImprovementReward,
    "component_relative_improvement": ComponentRelativeImprovementReward,
    "relative_l2_improvement": RelativeL2ImprovementReward,
    "relative_l2_log_improvement": RelativeL2LogImprovementReward,
    "worst_component_relative_improvement": WorstComponentRelativeImprovementReward,
    "component_balance_penalty": ComponentBalancePenaltyReward,
    "baseline_gap": BaselineGapReward,
    "normalized_baseline_gap": NormalizedBaselineGapReward,
    "log_normalized_baseline_gap": LogNormalizedBaselineGapReward,
    "normalized_baseline_gap_delta": NormalizedBaselineGapDeltaReward,
    "component_baseline_gap": ComponentBaselineGapReward,
    "worst_component_baseline_gap": WorstComponentBaselineGapReward,
    "relative_l2_baseline_gap": RelativeL2BaselineGapReward,
    "relative_l2_baseline_gap_delta": RelativeL2BaselineGapDeltaReward,
    "loss_l2_hybrid": LossL2HybridReward,
    "loss_l2_self_hybrid": LossL2SelfHybridReward,
    "running_best_l2_reward": RunningBestL2Reward,
    "progressive_loss_l2_hybrid": ProgressiveLossL2HybridReward,
}


def make_reward(name: str, params: dict | None = None) -> Reward:
    key = name.lower()
    if key not in REWARD_REGISTRY:
        known = ", ".join(sorted(REWARD_REGISTRY))
        raise ValueError(f"Unknown reward {name!r}. Known rewards: {known}")
    kwargs = dict(params or {})
    return REWARD_REGISTRY[key](**kwargs)
