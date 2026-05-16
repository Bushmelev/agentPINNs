from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .base import ConstraintSpec, EquationSpec, SampleBatch, gradients


def build_advection(
    beta: float = 1.0,
    data_path: str | None = None,
    sample_id: int = 0,
    target_time: float | None = None,
    time_tolerance: float = 1e-10,
) -> EquationSpec:
    if data_path is None:
        raise ValueError("advection requires equation_params.data_path")
    return build_advection_hdf5(
        data_path=data_path,
        sample_id=sample_id,
        target_time=target_time,
        time_tolerance=time_tolerance,
        beta=beta,
    )


def build_advection_hdf5(
    *,
    data_path: str,
    sample_id: int = 0,
    target_time: float | None = None,
    time_tolerance: float = 1e-10,
    beta: float = 1.0,
) -> EquationSpec:
    data = _load_advection_hdf5(data_path, sample_id)
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
    bc_t = t[1:]
    if pde_x.size == 0 or pde_t.size == 0:
        raise ValueError("HDF5 Advection grid must contain at least 3 x-points and 2 t-points")

    def residual(model: nn.Module, xt: torch.Tensor) -> torch.Tensor:
        xt = xt.clone().detach().requires_grad_(True)
        u = model(xt)
        grads = gradients(u, xt)
        u_x = grads[:, 0:1]
        u_t = grads[:, 1:2]
        return u_t + beta * u_x

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
        t_t = _column_tensor(bc_t, device)
        x_left = _column_tensor(np.full_like(bc_t, x[0], dtype=np.float64), device)
        x_right = _column_tensor(np.full_like(bc_t, x[-1], dtype=np.float64), device)
        left_xt = torch.cat([x_left, t_t], dim=1)
        right_xt = torch.cat([x_right, t_t], dim=1)
        return SampleBatch(xt=left_xt, x=x_left, t=t_t, aux=right_xt)

    def data_target(batch: SampleBatch, spec: EquationSpec) -> torch.Tensor:
        del spec
        if batch.y is None:
            raise ValueError("HDF5 target batch is missing y values")
        return batch.y

    def periodic_bc_loss(
        model: nn.Module,
        batch: SampleBatch,
        spec: EquationSpec,
    ) -> torch.Tensor:
        del spec
        if batch.aux is None:
            raise ValueError("Periodic boundary batch is missing right-boundary points")
        left = model(batch.xt)
        right = model(batch.aux)
        return torch.mean((left - right) ** 2)

    def reference_solver(
        spec: EquationSpec,
        nx: int = 201,
        nt: int = 201,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del spec, nx, nt
        return x.copy(), t.copy(), z.T.copy()

    return EquationSpec(
        name="advection",
        x_min=float(x[0]),
        x_max=float(x[-1]),
        t_min=float(t[0]),
        t_max=float(t[-1]),
        residual_fn=residual,
        pde_sampler=pde_sampler,
        constraints=(
            ConstraintSpec("ic", ic_sampler, data_target),
            ConstraintSpec("bc", bc_sampler, data_target, loss_fn=periodic_bc_loss),
        ),
        params={
            "beta": float(beta),
            "data_path": str(Path(data_path)),
            "sample_id": int(sample_id),
            "target_time": float(t[ic_index]),
        },
        default_batch_sizes={
            "pde": int(pde_x.size * pde_t.size),
            "ic": int(x.size),
            "bc": int(bc_t.size),
        },
        data_info={
            "source": "hdf5",
            "bc_type": "periodic",
            "path": str(Path(data_path)),
            "sample_id": int(sample_id),
            "tensor_shape": list(z.shape),
            "x_points": int(x.size),
            "t_points": int(t.size),
            "pde_points": int(pde_x.size * pde_t.size),
            "ic_points": int(x.size),
            "bc_points": int(bc_t.size),
            "ic_time_index": int(ic_index),
            "ic_time": float(t[ic_index]),
            "beta": float(beta),
        },
        reference_solver=reference_solver,
    )


def _load_advection_hdf5(path: str, sample_id: int) -> dict[str, np.ndarray]:
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Advection HDF5 file not found: {data_path}. "
            "Set equation_params.data_path to the correct file."
        )

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "h5py is required for HDF5 Advection data. Install dependencies with "
            "`pip install -e .`."
        ) from exc

    with h5py.File(data_path, "r") as file:
        z = np.asarray(file["tensor"][sample_id], dtype=np.float32)
        z = np.squeeze(z)
        t = np.asarray(_read_dataset(file, ("t-coordinate", "t")), dtype=np.float64)
        x = np.asarray(_read_dataset(file, ("x-coordinate", "x")), dtype=np.float64)

    if z.ndim != 2:
        raise ValueError(
            f"Expected 2D tensor sample after squeeze, got shape {z.shape} "
            f"for sample_id={sample_id}"
        )
    if z.shape == (t.size, x.size):
        pass
    elif z.shape == (x.size, t.size):
        z = z.T
    else:
        raise ValueError(
            f"Expected tensor sample shape {(t.size, x.size)} or {(x.size, t.size)}, "
            f"got {z.shape} for sample_id={sample_id}"
        )
    if x.size < 3 or t.size < 2:
        raise ValueError("HDF5 Advection data must contain at least 3 x-points and 2 t-points")
    return {"x": x, "t": t, "z": z}


def _read_dataset(file, names: tuple[str, ...]) -> np.ndarray:
    for name in names:
        if name in file:
            return file[name][:]
    known = ", ".join(file.keys())
    expected = ", ".join(names)
    raise KeyError(f"HDF5 file is missing one of [{expected}]. Known keys: {known}")


def _resolve_time_index(t: np.ndarray, target_time: float, tolerance: float) -> int:
    matches = np.where(np.abs(t - target_time) < tolerance)[0]
    if matches.size:
        return int(matches[0])
    return int(np.argmin(np.abs(t - target_time)))


def _column_tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32, device=device).reshape(-1, 1)


def _grid_batch(x: np.ndarray, t: np.ndarray, device: torch.device) -> SampleBatch:
    tt, xx = np.meshgrid(t, x, indexing="ij")
    x_t = _column_tensor(xx.reshape(-1), device)
    t_t = _column_tensor(tt.reshape(-1), device)
    return SampleBatch(xt=torch.cat([x_t, t_t], dim=1), x=x_t, t=t_t)
