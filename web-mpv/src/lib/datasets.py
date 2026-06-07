"""Datasets (CSV) stored under data/datasets/. Each dataset has columns FEATURE_KEYS + OUTPUT_KEYS."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import FEATURE_KEYS, OUTPUT_KEYS
from .storage import DATASETS_DIR


def list_datasets() -> list[dict]:
    out = []
    if not DATASETS_DIR.exists():
        return out
    for p in sorted(DATASETS_DIR.glob("*.csv")):
        try:
            n = sum(1 for _ in p.open()) - 1
        except OSError:
            n = 0
        out.append({"id": p.stem, "path": str(p), "n_rows": max(n, 0)})
    return out


def load_dataset(dataset_id: str) -> pd.DataFrame:
    return pd.read_csv(DATASETS_DIR / f"{dataset_id}.csv")


def save_dataset(dataset_id: str, df: pd.DataFrame) -> Path:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    path = DATASETS_DIR / f"{dataset_id}.csv"
    df.to_csv(path, index=False)
    return path


def empty_dataset() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_KEYS + OUTPUT_KEYS)


def create_dataset(dataset_id: str) -> Path:
    """Create an empty dataset file with the canonical columns."""
    return save_dataset(dataset_id, empty_dataset())


def rename_dataset(old_id: str, new_id: str) -> Path:
    src = DATASETS_DIR / f"{old_id}.csv"
    dst = DATASETS_DIR / f"{new_id}.csv"
    if not src.exists():
        raise FileNotFoundError(f"dataset `{old_id}` does not exist")
    if dst.exists():
        raise FileExistsError(f"dataset `{new_id}` already exists")
    src.rename(dst)
    return dst


def delete_dataset(dataset_id: str) -> None:
    path = DATASETS_DIR / f"{dataset_id}.csv"
    if path.exists():
        path.unlink()


def next_default_name(existing_ids: list[str]) -> str:
    n = 1
    while f"dataset_{n}" in existing_ids:
        n += 1
    return f"dataset_{n}"
