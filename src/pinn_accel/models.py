from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def get_activation(name: str) -> nn.Module:
    value = name.lower()
    if value == "tanh":
        return nn.Tanh()
    if value == "relu":
        return nn.ReLU()
    if value == "gelu":
        return nn.GELU()
    if value == "sine":
        return Sine()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(self, layers: Sequence[int], activation: str = "tanh"):
        super().__init__()
        if len(layers) < 2:
            raise ValueError("layers must contain input and output sizes")
        modules: list[nn.Module] = []
        for idx in range(len(layers) - 2):
            modules.append(nn.Linear(layers[idx], layers[idx + 1]))
            modules.append(get_activation(activation))
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)
