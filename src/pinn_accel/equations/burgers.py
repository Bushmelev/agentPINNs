from __future__ import annotations

import math
from pathlib import Path

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


def build_burgers(
    nu: float = 0.01 / math.pi,
    data_path: str | None = None,
    sample_id: int = 0,
    target_time: float | None = None,
    time_tolerance: float = 1e-10,
) -> EquationSpec:
    if data_path is not None:
        return build_burgers_hdf5(
            data_path=data_path,
            sample_id=sample_id,
            target_time=target_time,
            time_tolerance=time_tolerance,
            nu=nu,
        )

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


def build_burgers_hdf5(
    *,
    data_path: str,
    sample_id: int = 0,
    target_time: float | None = None,
    time_tolerance: float = 1e-10,
    nu: float = 0.001,
) -> EquationSpec:
    data = _load_burgers_hdf5(data_path, sample_id)
    x = data["x"]
    t = data["t"]
    z = data["z"]
    ic_index = _resolve_time_index(
        t,
        t[0] if target_time is None else target_time,
        time_tolerance,
    )
    pde_x = x[1:-1]
    pde_t = t[1:]
    if pde_x.size == 0 or pde_t.size == 0:
        raise ValueError("HDF5 Burgers grid must contain at least 3 x-points and 2 t-points")

    def residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        xt = xt.clone().detach().requires_grad_(True)
        u = model(xt)
        grads = gradients(u, xt)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        u_xx = gradients(u_x, xt)[:, 0:1]
        return u_t + u * u_x - nu * u_xx

    def pde_sampler(
        n: int,
        spec: EquationSpec,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> SampleBatch:
        del n, spec, generator
        return _grid_batch(pde_x, pde_t, device)

    def ic_sampler(
        n: int,
        spec: EquationSpec,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> SampleBatch:
        del n, spec, generator
        x_t = _column_tensor(x, device)
        t_t = torch.full_like(x_t, float(t[ic_index]))
        y_t = _column_tensor(z[ic_index, :], device)
        return SampleBatch(xt=torch.cat([x_t, t_t], dim=1), x=x_t, t=t_t, y=y_t)

    def bc_sampler(
        n: int,
        spec: EquationSpec,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> SampleBatch:
        del n, spec, generator
        t_t = _column_tensor(np.concatenate([t, t]), device)
        x_t = _column_tensor(
            np.concatenate(
                [
                    np.full_like(t, x[0], dtype=np.float64),
                    np.full_like(t, x[-1], dtype=np.float64),
                ]
            ),
            device,
        )
        y_t = _column_tensor(np.concatenate([z[:, 0], z[:, -1]]), device)
        return SampleBatch(xt=torch.cat([x_t, t_t], dim=1), x=x_t, t=t_t, y=y_t)

    def data_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        if batch.y is None:
            raise ValueError("HDF5 target batch is missing y values")
        return batch.y

    def reference_solver(
        spec: EquationSpec,
        nx: int = 201,
        nt: int = 201,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del spec, nx, nt
        return x.copy(), t.copy(), z.T.copy()

    return EquationSpec(
        name="burgers",
        x_min=float(x[0]),
        x_max=float(x[-1]),
        t_min=float(t[0]),
        t_max=float(t[-1]),
        residual_fn=residual,
        pde_sampler=pde_sampler,
        constraints=(
            ConstraintSpec("ic", ic_sampler, data_target),
            ConstraintSpec("bc", bc_sampler, data_target),
        ),
        params={
            "nu": float(nu),
            "data_path": str(Path(data_path)),
            "sample_id": int(sample_id),
            "target_time": float(t[ic_index]),
        },
        default_batch_sizes={
            "pde": int(pde_x.size * pde_t.size),
            "ic": int(x.size),
            "bc": int(2 * t.size),
        },
        data_info={
            "source": "hdf5",
            "path": str(Path(data_path)),
            "sample_id": int(sample_id),
            "tensor_shape": list(z.shape),
            "x_points": int(x.size),
            "t_points": int(t.size),
            "pde_points": int(pde_x.size * pde_t.size),
            "ic_points": int(x.size),
            "bc_points": int(2 * t.size),
            "ic_time_index": int(ic_index),
            "ic_time": float(t[ic_index]),
            "nu": float(nu),
        },
        reference_solver=reference_solver,
    )


def _load_burgers_hdf5(path: str, sample_id: int) -> dict[str, np.ndarray]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Burgers HDF5 file not found: {data_path}. "
            "Set equation_params.data_path to the correct file."
        )

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required for HDF5 Burgers data. Install dependencies with "
            "`pip install -e .`."
        ) from exc

    with h5py.File(data_path, "r") as file:
        z = np.asarray(file["tensor"][sample_id], dtype=np.float32)
        t = np.asarray(file["t-coordinate"][:], dtype=np.float64)
        x = np.asarray(file["x-coordinate"][:], dtype=np.float64)

    expected = (t.size, x.size)
    if z.shape != expected:
        raise ValueError(
            f"Expected tensor sample shape {expected}, got {z.shape} for sample_id={sample_id}"
        )
    if x.size < 3 or t.size < 2:
        raise ValueError("HDF5 Burgers data must contain at least 3 x-points and 2 t-points")
    return {"x": x, "t": t, "z": z}


def _resolve_time_index(t: np.ndarray, target_time: float, tolerance: float) -> int:
    matches = np.where(np.abs(t - target_time) < tolerance)[0]
    if matches.size:
        return int(matches[0])
    closest = int(np.argmin(np.abs(t - target_time)))
    return closest


def _column_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)


def _grid_batch(x: np.ndarray, t: np.ndarray, device: torch.device) -> SampleBatch:
    tt, xx = np.meshgrid(t, x, indexing="ij")
    x_t = _column_tensor(xx.reshape(-1), device)
    t_t = _column_tensor(tt.reshape(-1), device)
    return SampleBatch(xt=torch.cat([x_t, t_t], dim=1), x=x_t, t=t_t)


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
