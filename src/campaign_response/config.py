"""Load the YAML config and resolve paths against the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def path(self, section: str, key: str) -> Path:
        p = ROOT / self.raw[section][key]
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def raw_file(self) -> Path:
        return ROOT / self.raw["data"]["raw_file"]

    @property
    def clean_file(self) -> Path:
        p = ROOT / self.raw["data"]["processed_file"]
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports(self) -> Path:
        return self.path("outputs", "reports_dir")

    @property
    def figures(self) -> Path:
        from .plotting import SLIDE_MODE

        f = self.path("outputs", "figures_dir")
        if SLIDE_MODE:
            f = f / "slides"
            f.mkdir(exist_ok=True)
        return f

    @property
    def models(self) -> Path:
        return self.path("outputs", "models_dir")


def load_config(path: str | Path = ROOT / "config.yaml") -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))
