from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch


def config_to_dict(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


def build_agent_checkpoint(controller: Any) -> dict[str, Any] | None:
    agent = getattr(controller, "agent", None)
    if agent is None:
        return None

    policy = getattr(agent, "policy", None)
    agent_class = type(agent).__name__
    feature_order = []
    if getattr(agent, "include_log_losses", False):
        feature_order.append("log_losses")
    feature_order.extend(["dlog_losses", "log_lambdas"])
    if getattr(agent, "include_initial_loss_ratios", False):
        feature_order.append("log_initial_loss_ratios")
    feature_order.append("progress")

    payload: dict[str, Any] = {
        "agent_class": agent_class,
        "agent_registry_name": (
            "tiny_loss_weight" if agent_class == "TinyLossWeightAgent" else None
        ),
        "controller_class": type(controller).__name__,
        "controller_name": getattr(controller, "name", None),
        "reward_name": getattr(getattr(controller, "reward", None), "name", None),
        "reward_requires_baseline": getattr(
            getattr(controller, "reward", None),
            "requires_baseline",
            None,
        ),
        "component_names": list(getattr(agent, "component_names", [])),
        "state_dim": agent.state_dim(),
        "action_dim": getattr(agent, "action_dim", None),
        "state_features": {
            "feature_order": feature_order,
            "include_log_losses": getattr(agent, "include_log_losses", None),
            "include_initial_loss_ratios": getattr(
                agent,
                "include_initial_loss_ratios",
                None,
            ),
            "feature_clip": getattr(agent, "feature_clip", None),
        },
        "action_transform": {
            "action_scale": getattr(agent, "action_scale", None),
            "min_weight": getattr(agent, "min_weight", None),
            "min_weight_share": getattr(agent, "min_weight_share", None),
            "max_weight_share": getattr(agent, "max_weight_share", None),
        },
        "policy_config": {
            "policy_class": type(policy).__name__ if policy is not None else None,
            "policy_bias": getattr(agent, "policy_bias", None),
            "policy_hidden_dim": getattr(agent, "policy_hidden_dim", None),
            "sigma": getattr(agent, "sigma", None),
            "learn_sigma": getattr(agent, "learn_sigma", None),
            "sigma_min": getattr(agent, "sigma_min", None),
            "sigma_max": getattr(agent, "sigma_max", None),
            "zero_init_policy": getattr(agent, "zero_init_policy", None),
        },
        "agent_state_dict": agent.state_dict(),
    }

    if policy is not None:
        payload["policy_state_dict"] = policy.state_dict()
    current_sigma = getattr(agent, "current_sigma", None)
    if current_sigma is not None:
        payload["current_sigma"] = current_sigma()
    optimizer = getattr(agent, "optimizer", None)
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def load_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        payload = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Checkpoint payload must be a dict: {checkpoint_path}")
    return payload


def extract_agent_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    agent_payload = payload.get("agent")
    if isinstance(agent_payload, dict):
        return agent_payload
    if "policy_state_dict" in payload or "agent_state_dict" in payload:
        return payload
    raise ValueError("Checkpoint does not contain an agent payload")


def load_agent_checkpoint_payload(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    return extract_agent_checkpoint(
        load_checkpoint_payload(path, map_location=map_location)
    )


def agent_init_kwargs_from_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    state_features = payload.get("state_features", {})
    if isinstance(state_features, dict):
        for key in (
            "include_log_losses",
            "include_initial_loss_ratios",
            "feature_clip",
        ):
            if key in state_features and state_features[key] is not None:
                kwargs[key] = state_features[key]

    action_transform = payload.get("action_transform", {})
    if isinstance(action_transform, dict):
        for key in (
            "action_scale",
            "min_weight",
            "min_weight_share",
            "max_weight_share",
        ):
            if key in action_transform and action_transform[key] is not None:
                kwargs[key] = action_transform[key]

    policy_config = payload.get("policy_config", {})
    if isinstance(policy_config, dict):
        for key in (
            "policy_bias",
            "policy_hidden_dim",
            "sigma",
            "learn_sigma",
            "sigma_min",
            "sigma_max",
            "zero_init_policy",
        ):
            if key in policy_config:
                kwargs[key] = policy_config[key]
    return kwargs


def build_result_checkpoint(
    *,
    equation_name: str,
    label: str,
    result: Any,
    model_config: Any,
    training_config: Any,
) -> dict[str, Any]:
    model_state = result.model.state_dict()
    payload: dict[str, Any] = {
        "equation": equation_name,
        "controller": label,
        "model_config": config_to_dict(model_config),
        "training_config": config_to_dict(training_config),
        "state_dict": model_state,
        "model_state_dict": model_state,
        "controller_class": type(result.controller).__name__,
        "controller_name": getattr(result.controller, "name", None),
        "component_names": list(getattr(result.controller, "component_names", [])),
        "controller_state_dict": result.controller.state_dict(),
        "history": result.history,
    }
    agent_checkpoint = build_agent_checkpoint(result.controller)
    if agent_checkpoint is not None:
        payload["agent"] = agent_checkpoint
    return payload
