from __future__ import annotations

from typing import Any

from .actor_critic import ActorCriticAgent
from .base import BaseWeightAgent
from .policy_gradient import PolicyGradientAgent


def make_agent(name: str, params: dict[str, Any]) -> BaseWeightAgent:
    value = name.lower()
    agent_params = dict(params)
    agent_params.pop("reward", None)
    hidden_sizes = agent_params.get("hidden_sizes")
    if hidden_sizes is not None:
        agent_params["hidden_sizes"] = tuple(hidden_sizes)
    if value == "policy_gradient":
        return PolicyGradientAgent(**agent_params)
    if value == "actor_critic":
        return ActorCriticAgent(**agent_params)
    raise ValueError(f"Unknown agent: {name}")


__all__ = [
    "ActorCriticAgent",
    "BaseWeightAgent",
    "PolicyGradientAgent",
    "make_agent",
]
