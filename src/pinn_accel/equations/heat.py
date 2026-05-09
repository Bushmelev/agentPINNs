from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .base import (
    ConstraintSpec,
    EquationSpec,
    SampleBatch,
    gradients,
    sample_boundary,
    sample_initial,
    target_to_numpy,
)


def build_heat(alpha: float = 0.1) -> EquationSpec:
    def residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        xt = xt.clone().detach().requires_grad_(True)
        u = model(xt)
        grads = gradients(u, xt)
        u_t = grads[:, 1:2]
        u_x = grads[:, 0:1]
        u_xx = gradients(u_x, xt)[:, 0:1]
        return u_t - alpha * u_xx

    def ic_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        return torch.sin(math.pi * batch.x)

    def bc_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        return torch.zeros_like(batch.x)

    def reference_solver(
        spec: EquationSpec,
        nx: int = 201,
        nt: int = 201,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return solve_heat_fdm(spec, ic_target, bc_target, nx=nx, nt=nt, alpha=alpha)

    return EquationSpec(
        name="heat",
        x_min=-1.0,
        x_max=1.0,
        t_min=0.0,
        t_max=1.0,
        residual_fn=residual,
        constraints=(
            ConstraintSpec("ic", sample_initial, ic_target),
            ConstraintSpec("bc", sample_boundary, bc_target),
        ),
        params={"alpha": float(alpha)},
        reference_solver=reference_solver,
    )


def solve_heat_fdm(
    spec: EquationSpec,
    ic_target,
    bc_target,
    nx: int,
    nt: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    x = np.linspace(spec.x_min, spec.x_max, nx)
    dx = x[1] - x[0]
    dt = (spec.t_max - spec.t_min) / float(nt - 1)
    r = alpha * dt / dx**2
    if r > 0.5:
        nt = math.ceil(alpha * (spec.t_max - spec.t_min) / (0.5 * dx**2)) + 1
        dt = (spec.t_max - spec.t_min) / float(nt - 1)
        r = alpha * dt / dx**2

    t = np.linspace(spec.t_min, spec.t_max, nt)
    u = np.zeros((nx, nt), dtype=np.float64)
    u[:, 0] = target_to_numpy(ic_target, spec, x, np.full_like(x, spec.t_min), device)
    left = target_to_numpy(bc_target, spec, np.full_like(t, spec.x_min), t, device)
    right = target_to_numpy(bc_target, spec, np.full_like(t, spec.x_max), t, device)
    u[0, :] = left
    u[-1, :] = right

    for step in range(nt - 1):
        u[1:-1, step + 1] = u[1:-1, step] + r * (
            u[2:, step] - 2.0 * u[1:-1, step] + u[:-2, step]
        )
        u[0, step + 1] = left[step + 1]
        u[-1, step + 1] = right[step + 1]
    return x, t, u
