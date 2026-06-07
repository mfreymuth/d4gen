"""Models registry: list/load/save sklearn models on disk via joblib.

A model lives under data/models/{shared,mine}/<model_id>/:
  - <output_key>.joblib   (one regressor per output)
  - meta.json             (name, date, metrics, dataset_id, ...)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
from xgboost import XGBRegressor

from .schema import OUTPUT_KEYS
from .storage import read_json, scope_dir, write_json

Scope = Literal["shared", "mine"]


def _model_dir(scope: Scope, model_id: str) -> Path:
    return scope_dir(scope) / model_id


def list_models(scope: Scope) -> list[dict]:
    root = scope_dir(scope)
    if not root.exists():
        return []
    out = []
    for d in root.iterdir():
        meta_path = d / "meta.json"
        if d.is_dir() and meta_path.exists():
            out.append(read_json(meta_path))
    out.sort(key=lambda m: m.get("date", ""), reverse=True)
    return out


def load_model(scope: Scope, model_id: str) -> dict[str, XGBRegressor]:
    d = _model_dir(scope, model_id)
    return {key: joblib.load(d / f"{key}.joblib") for key in OUTPUT_KEYS}


def save_model(
    scope: Scope,
    name: str,
    models: dict[str, XGBRegressor],
    metrics: dict,
    dataset_id: str,
) -> str:
    model_id = uuid.uuid4().hex[:10]
    d = _model_dir(scope, model_id)
    d.mkdir(parents=True, exist_ok=True)

    total_size = 0
    for key, m in models.items():
        path = d / f"{key}.joblib"
        joblib.dump(m, path)
        total_size += path.stat().st_size

    meta = {
        "id": model_id,
        "name": name,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset_id": dataset_id,
        "size_bytes": total_size,
        **metrics,
    }
    write_json(d / "meta.json", meta)
    return model_id


def latest(scope: Scope) -> dict | None:
    items = list_models(scope)
    return items[0] if items else None
