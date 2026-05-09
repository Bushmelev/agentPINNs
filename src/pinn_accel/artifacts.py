from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass(frozen=True)
class ArtifactStore:
    root: Path

    @classmethod
    def create(cls, base_dir: str | Path) -> "ArtifactStore":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = Path(base_dir) / timestamp
        root.mkdir(parents=True, exist_ok=False)
        return cls(root=root)

    def equation_dir(self, equation_name: str) -> Path:
        path = self.root / slugify(equation_name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def method_dir(self, equation_name: str, label: str) -> Path:
        path = self.equation_dir(equation_name) / slugify(label)
        (path / "plots").mkdir(parents=True, exist_ok=True)
        return path

    def save_json(self, path: str | Path, payload: Any) -> None:
        full_path = Path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(
            json.dumps(to_jsonable(payload), indent=2),
            encoding="utf-8",
        )

    def save_history(self, equation_name: str, label: str, history: dict[str, Any]) -> Path:
        path = self.method_dir(equation_name, label) / "history.json"
        self.save_json(path, history)
        return path

    def save_checkpoint(
        self,
        equation_name: str,
        label: str,
        payload: dict[str, Any],
    ) -> Path:
        path = self.method_dir(equation_name, label) / "checkpoint.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return path
