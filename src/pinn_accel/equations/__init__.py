from __future__ import annotations

from collections.abc import Callable

from .base import ConstraintSpec, EquationSpec, SampleBatch
from .advection import build_advection
from .burgers import build_burgers
from .heat import build_heat


EquationBuilder = Callable[..., EquationSpec]

EQUATION_REGISTRY: dict[str, EquationBuilder] = {
    "advection": build_advection,
    "burgers": build_burgers,
    "heat": build_heat,
}


def get_equation(name: str, **kwargs) -> EquationSpec:
    key = name.lower()
    if key not in EQUATION_REGISTRY:
        known = ", ".join(sorted(EQUATION_REGISTRY))
        raise KeyError(f"Unknown equation {name!r}. Known equations: {known}")
    return EQUATION_REGISTRY[key](**kwargs)


__all__ = [
    "ConstraintSpec",
    "EquationSpec",
    "SampleBatch",
    "EQUATION_REGISTRY",
    "build_advection",
    "get_equation",
]
