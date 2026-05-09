from __future__ import annotations

from collections.abc import Callable
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
            return self.sampler(self.sample_size, self.spec, self.device, generator=self.generator)
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
        )


@dataclass
class LossPack:
    names: list[str]
    values: torch.Tensor
    by_name: dict[str, torch.Tensor]

    def scalar_dict(self) -> dict[str, float]:
        return {name: float(value.detach().cpu().item()) for name, value in self.by_name.items()}


class LossEvaluator:
    def __init__(
        self,
        spec: EquationSpec,
        batch_sizes: dict[str, int],
        pool_sizes: dict[str, int],
        device: torch.device,
        seed: int,
    ):
        self.spec = spec
        self.device = device
        self.component_names = spec.component_names
        self.samplers: dict[str, ComponentSampler] = {}
        self._constraint_by_name = {constraint.name: constraint for constraint in spec.constraints}

        for idx, name in enumerate(self.component_names):
            if name not in batch_sizes:
                raise KeyError(f"batch_sizes must contain {name!r}")
            sampler = sample_interior if name == "pde" else self._constraint_by_name[name].sampler
            self.samplers[name] = ComponentSampler.build(
                name=name,
                sampler=sampler,
                sample_size=int(batch_sizes[name]),
                pool_size=pool_sizes.get(name),
                spec=spec,
                device=device,
                seed=seed + 7919 * (idx + 1),
            )

    def compute(self, model: nn.Module) -> LossPack:
        by_name: dict[str, torch.Tensor] = {}
        pde_batch = self.samplers["pde"].draw()
        by_name["pde"] = residual_mse(self.spec.residual(model, pde_batch.xt))

        for constraint in self.spec.constraints:
            batch = self.samplers[constraint.name].draw()
            prediction = model(batch.xt)
            target = constraint.target_fn(batch, self.spec)
            by_name[constraint.name] = mse(prediction, target)

        values = torch.stack([by_name[name] for name in self.component_names])
        return LossPack(names=list(self.component_names), values=values, by_name=by_name)
