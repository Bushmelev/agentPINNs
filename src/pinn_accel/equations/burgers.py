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


def build_burgers(nu: float = 0.01 / math.pi) -> EquationSpec:
    def residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        xt = xt.clone().detach().requires_grad_(True)
        u = model(xt)
        grads = gradients(u, xt)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = gradients(u_x, xt)[:, 0:1]
        return u_t + u * u_x - nu * u_xx

    def ic_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        return -torch.sin(math.pi * batch.x)

    def bc_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        return torch.zeros_like(batch.x)

    def reference_solver(
        spec: EquationSpec,
        nx: int = 201,
        nt: int = 201,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return solve_burgers_fdm(spec, ic_target, bc_target, nx=nx, nt=nt, nu=nu)

    return EquationSpec(
        name="burgers",
        x_min=-1.0,
        x_max=1.0,
        t_min=0.0,
        t_max=1.0,
        residual_fn=residual,
        constraints=(
            ConstraintSpec("ic", sample_initial, ic_target),
            ConstraintSpec("bc", sample_boundary, bc_target),
        ),
        params={"nu": float(nu)},
        reference_solver=reference_solver,
    )


def solve_burgers_fdm(
    spec: EquationSpec,
    ic_target,
    bc_target,
    nx: int,
    nt: int,
    nu: float,
    cfl: float = 0.4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = torch.device("cpu")
    x = np.linspace(spec.x_min, spec.x_max, nx)
    dx = x[1] - x[0]
    u0 = target_to_numpy(ic_target, spec, x=x, t=np.full_like(x, spec.t_min), device=device)
    max_u = max(float(np.max(np.abs(u0))), 1e-6)
    dt_adv = cfl * dx / max_u
    dt_diff = cfl * dx**2 / (2.0 * nu) if nu > 0 else dt_adv
    dt_max = min(dt_adv, dt_diff)
    requested_dt = (spec.t_max - spec.t_min) / float(nt - 1)
    if requested_dt > dt_max:
        nt = math.ceil((spec.t_max - spec.t_min) / dt_max) + 1

    t = np.linspace(spec.t_min, spec.t_max, nt)
    dt = t[1] - t[0]
    u = np.zeros((nx, nt), dtype=np.float64)
    u[:, 0] = u0
    left = target_to_numpy(bc_target, spec, np.full_like(t, spec.x_min), t, device)
    right = target_to_numpy(bc_target, spec, np.full_like(t, spec.x_max), t, device)
    u[0, :] = left
    u[-1, :] = right

    for step in range(nt - 1):
        u_n = u[:, step].copy()
        flux = 0.5 * u_n**2
        speed = np.maximum(np.abs(u_n[:-1]), np.abs(u_n[1:]))
        edge_flux = 0.5 * (flux[:-1] + flux[1:]) - 0.5 * speed * (u_n[1:] - u_n[:-1])
        u[1:-1, step + 1] = (
            u_n[1:-1]
            - (dt / dx) * (edge_flux[1:] - edge_flux[:-1])
            + nu * dt / dx**2 * (u_n[2:] - 2.0 * u_n[1:-1] + u_n[:-2])
        )
        u[0, step + 1] = left[step + 1]
        u[-1, step + 1] = right[step + 1]
        if not np.isfinite(u[:, step + 1]).all():
            raise ValueError("Burgers reference solver became unstable")
    return x, t, u
