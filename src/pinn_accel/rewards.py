from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardContext:
    initial_total: float
    previous_total: float
    current_total: float
    previous_losses: np.ndarray
    current_losses: np.ndarray
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


class RelativeL2ImprovementReward(Reward):
    name = "relative_l2_improvement"

    def __call__(self, ctx: RewardContext) -> float:
        _, previous, current = _require_relative_l2(ctx)
        eps = 1e-8
        return float((previous - current) / (abs(previous) + eps))


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
        if ctx.baseline_initial_total is None or ctx.baseline_current_total is None:
            raise ValueError("normalized_baseline_gap reward requires baseline totals")
        eps = 1e-8
        baseline_norm = ctx.baseline_current_total / (ctx.baseline_initial_total + eps)
        agent_norm = ctx.current_total / (ctx.initial_total + eps)
        return float(baseline_norm - agent_norm)


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
        initial, _, current = _require_relative_l2(ctx)
        baseline_initial, _, baseline_current = _require_baseline_relative_l2(ctx)
        eps = 1e-8
        baseline_norm = baseline_current / (baseline_initial + eps)
        agent_norm = current / (initial + eps)
        return float(baseline_norm - agent_norm)


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


REWARD_REGISTRY = {
    "log_ratio": LogRatioReward,
    "relative_improvement": RelativeImprovementReward,
    "component_relative_improvement": ComponentRelativeImprovementReward,
    "relative_l2_improvement": RelativeL2ImprovementReward,
    "baseline_gap": BaselineGapReward,
    "normalized_baseline_gap": NormalizedBaselineGapReward,
    "normalized_baseline_gap_delta": NormalizedBaselineGapDeltaReward,
    "relative_l2_baseline_gap": RelativeL2BaselineGapReward,
    "relative_l2_baseline_gap_delta": RelativeL2BaselineGapDeltaReward,
}


def make_reward(name: str, params: dict | None = None) -> Reward:
    key = name.lower()
    if key not in REWARD_REGISTRY:
        known = ", ".join(sorted(REWARD_REGISTRY))
        raise ValueError(f"Unknown reward {name!r}. Known rewards: {known}")
    kwargs = dict(params or {})
    return REWARD_REGISTRY[key](**kwargs)
