from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


ReferenceSolution = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class SampleBatch:
    xt: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor
    y: torch.Tensor | None = None
    aux: torch.Tensor | None = None


TargetFn = Callable[[SampleBatch, "EquationSpec"], torch.Tensor]
ConstraintLossFn = Callable[[nn.Module, SampleBatch, "EquationSpec"], torch.Tensor]
Sampler = Callable[..., SampleBatch]
ResidualFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]
ReferenceSolver = Callable[["EquationSpec", int, int], ReferenceSolution]


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    sampler: Sampler
    target_fn: TargetFn
    loss_fn: ConstraintLossFn | None = None


@dataclass(frozen=True)
class EquationSpec:
    name: str
    x_min: float
    x_max: float
    t_min: float
    t_max: float
    residual_fn: ResidualFn
    constraints: Sequence[ConstraintSpec] = field(default_factory=tuple)
    pde_sampler: Sampler | None = None
    params: dict[str, Any] = field(default_factory=dict)
    default_batch_sizes: dict[str, int] = field(default_factory=dict)
    data_info: dict[str, Any] = field(default_factory=dict)
    reference_solver: ReferenceSolver | None = None

    @property
    def component_names(self) -> list[str]:
        return ["pde", *[constraint.name for constraint in self.constraints]]

    def residual(self, model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        return self.residual_fn(model, xt)

    def solve_reference(self, nx: int = 201, nt: int = 201) -> ReferenceSolution:
        if self.reference_solver is None:
            raise ValueError(f"{self.name} has no reference solver")
        return self.reference_solver(self, nx, nt)


def gradients(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


def _rand_uniform(
    n: int,
    low: float,
    high: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    values = torch.rand(n, 1, generator=generator, dtype=torch.float32)
    values = values * (high - low) + low
    return values.to(device=device)


def sample_interior(
    n: int,
    spec: EquationSpec,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> SampleBatch:
    x = _rand_uniform(n, spec.x_min, spec.x_max, device, generator)
    t = _rand_uniform(n, spec.t_min, spec.t_max, device, generator)
    return SampleBatch(xt=torch.cat([x, t], dim=1), x=x, t=t)


def sample_initial(
    n: int,
    spec: EquationSpec,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> SampleBatch:
    x = _rand_uniform(n, spec.x_min, spec.x_max, device, generator)
    t = torch.full_like(x, spec.t_min)
    return SampleBatch(xt=torch.cat([x, t], dim=1), x=x, t=t)


def sample_boundary(
    n: int,
    spec: EquationSpec,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> SampleBatch:
    n_left = n // 2
    n_right = n - n_left
    t_left = _rand_uniform(n_left, spec.t_min, spec.t_max, device, generator)
    t_right = _rand_uniform(n_right, spec.t_min, spec.t_max, device, generator)
    x_left = torch.full_like(t_left, spec.x_min)
    x_right = torch.full_like(t_right, spec.x_max)
    x = torch.cat([x_left, x_right], dim=0)
    t = torch.cat([t_left, t_right], dim=0)
    return SampleBatch(xt=torch.cat([x, t], dim=1), x=x, t=t)


def target_to_numpy(
    target_fn: TargetFn,
    spec: EquationSpec,
    x: np.ndarray,
    t: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_t = torch.tensor(x, dtype=torch.float32, device=device).reshape(-1, 1)
    t_t = torch.tensor(t, dtype=torch.float32, device=device).reshape(-1, 1)
    batch = SampleBatch(xt=torch.cat([x_t, t_t], dim=1), x=x_t, t=t_t)
    with torch.no_grad():
        target = target_fn(batch, spec)
    return target.detach().cpu().numpy().reshape(-1)
