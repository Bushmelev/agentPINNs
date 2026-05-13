from __future__ import annotations

from typing import Any

from .base import BaseWeightAgent
from .tiny import LinearRLPolicy, TinyLossWeightAgent


AGENT_REGISTRY = {
    "tiny_loss_weight": TinyLossWeightAgent,
}
AGENT_NAMES = frozenset(AGENT_REGISTRY)


def make_agent(name: str, params: dict[str, Any]) -> BaseWeightAgent:
    value = name.lower()
    agent_params = dict(params)
    agent_params.pop("reward", None)
    if value in AGENT_REGISTRY:
        return AGENT_REGISTRY[value](**agent_params)
    raise ValueError(f"Unknown agent: {name}")


__all__ = [
    "AGENT_NAMES",
    "AGENT_REGISTRY",
    "BaseWeightAgent",
    "LinearRLPolicy",
    "TinyLossWeightAgent",
    "make_agent",
]
