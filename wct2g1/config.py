"""Config loading and hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"


class Config(dict):
    """Config dict that knows where it was loaded from and its content hash."""

    def __init__(self, data: dict[str, Any], path: Path):
        super().__init__(data)
        self.path = path
        self.root = path.parent
        self.hash = hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]

    def path_of(self, key: str) -> Path:
        """Resolve a key from the `paths` section to an absolute path."""
        p = Path(self["paths"][key])
        return p if p.is_absolute() else self.root / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as f:
        data = yaml.safe_load(f)
    return Config(data, path.resolve())
