from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn

from .equations.base import EquationSpec, SampleBatch, Sampler, sample_interior


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def residual_mse(residual: torch.Tensor) -> torch.Tensor:
    return torch.mean(residual**2)


def _seeded_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


@dataclass
class ComponentSampler:
    name: str
    sampler: Sampler
    sample_size: int
    spec: EquationSpec
    device: torch.device
    generator: torch.Generator
    pool: SampleBatch | None = None

    @classmethod
    def build(
        cls,
        *,
        name: str,
        sampler: Sampler,
        sample_size: int,
        pool_size: int | None,
        spec: EquationSpec,
        device: torch.device,
        seed: int,
    ) -> "ComponentSampler":
        if sample_size <= 0:
            raise ValueError(f"{name} sample_size must be positive, got {sample_size}")
        if pool_size is not None and pool_size < sample_size:
            raise ValueError(
                f"{name} pool_size={pool_size} is smaller than sample_size={sample_size}"
            )
        generator = _seeded_generator(seed)
        pool = None
        if pool_size is not None:
            pool = sampler(pool_size, spec, device, generator=_seeded_generator(seed + 1))
        return cls(name, sampler, sample_size, spec, device, generator, pool)

    def draw(self) -> SampleBatch:
        if self.pool is None:
            return self.sampler(
                self.sample_size,
                self.spec,
                self.device,
                generator=self.generator,
            )
        pool_size = self.pool.xt.shape[0]
        if self.sample_size == pool_size:
            return self.pool
        indices = torch.randperm(pool_size, generator=self.generator)[: self.sample_size].to(
            self.pool.xt.device
        )
        return SampleBatch(
            xt=self.pool.xt.index_select(0, indices),
            x=self.pool.x.index_select(0, indices),
            t=self.pool.t.index_select(0, indices),
            y=None if self.pool.y is None else self.pool.y.index_select(0, indices),
            aux=(
                None
                if self.pool.aux is None
                else self.pool.aux.index_select(0, indices)
            ),
        )

    def describe(self) -> dict[str, int | str | bool | None]:
        return {
            "sample_size": int(self.sample_size),
            "pool_size": None if self.pool is None else int(self.pool.xt.shape[0]),
            "uses_pool": self.pool is not None,
            "full_batch": self.pool is not None and self.sample_size == self.pool.xt.shape[0],
        }


@dataclass
class LossPack:
    names: list[str]
    values: torch.Tensor
    by_name: dict[str, torch.Tensor]

    def scalar_dict(self) -> dict[str, float]:
        return {
            name: float(value.detach().cpu().item())
            for name, value in self.by_name.items()
        }


class LossEvaluator:
    def __init__(
        self,
        spec: EquationSpec,
        batch_sizes: dict[str, int],
        pool_sizes: dict[str, int],
        device: torch.device,
        seed: int,
        full_batch: bool = True,
    ):
        self.spec = spec
        self.device = device
        self.component_names = spec.component_names
        self.samplers: dict[str, ComponentSampler] = {}
        self._constraint_by_name = {
            constraint.name: constraint
            for constraint in spec.constraints
        }
        self.full_batch = bool(full_batch)

        for idx, name in enumerate(self.component_names):
            default_size = spec.default_batch_sizes.get(name)
            if self.full_batch and default_size is not None:
                sample_size = int(default_size)
                pool_size = int(default_size)
            elif name in batch_sizes:
                sample_size = int(batch_sizes[name])
                pool_size = pool_sizes.get(name)
            elif default_size is not None:
                sample_size = int(default_size)
                pool_size = int(default_size)
            else:
                raise KeyError(f"batch_sizes must contain {name!r}")
            if name == "pde":
                sampler = spec.pde_sampler or sample_interior
            else:
                sampler = self._constraint_by_name[name].sampler
            self.samplers[name] = ComponentSampler.build(
                name=name,
                sampler=sampler,
                sample_size=sample_size,
                pool_size=pool_size,
                spec=spec,
                device=device,
                seed=seed + 7919 * (idx + 1),
            )
        self.batch_info = self._make_batch_info()

    def _make_batch_info(self) -> dict:
        return {
            "full_batch_requested": self.full_batch,
            "data_info": self.spec.data_info,
            "components": {
                name: sampler.describe()
                for name, sampler in self.samplers.items()
            },
        }

    def draw_batches(self) -> dict[str, SampleBatch]:
        batches = {"pde": self.samplers["pde"].draw()}
        for constraint in self.spec.constraints:
            batches[constraint.name] = self.samplers[constraint.name].draw()
        return batches

    def compute(
        self,
        model: nn.Module,
        batches: Mapping[str, SampleBatch] | None = None,
    ) -> LossPack:
        if batches is None:
            batches = self.draw_batches()
        by_name: dict[str, torch.Tensor] = {}
        pde_batch = batches["pde"]
        by_name["pde"] = residual_mse(self.spec.residual(model, pde_batch.xt))

        for constraint in self.spec.constraints:
            batch = batches[constraint.name]
            if constraint.loss_fn is not None:
                by_name[constraint.name] = constraint.loss_fn(model, batch, self.spec)
            else:
                prediction = model(batch.xt)
                target = constraint.target_fn(batch, self.spec)
                by_name[constraint.name] = mse(prediction, target)

        values = torch.stack([by_name[name] for name in self.component_names])
        return LossPack(names=list(self.component_names), values=values, by_name=by_name)
