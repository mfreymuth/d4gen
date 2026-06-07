"""Filesystem layout + thin I/O helpers. No DB."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = DATA / "models"
SHARED_DIR = MODELS / "shared"
MINE_DIR = MODELS / "mine"
DATASETS_DIR = DATA / "datasets"
SEED_DIR = DATA / "seed"
def ensure_dirs() -> None:
    for d in (SHARED_DIR, MINE_DIR, DATASETS_DIR, SEED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def scope_dir(scope: str) -> Path:
    if scope == "shared":
        return SHARED_DIR
    if scope == "mine":
        return MINE_DIR
    raise ValueError(f"unknown scope: {scope}")
