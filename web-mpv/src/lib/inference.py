"""Predict outputs + uncertainty intervals.

Two model formats are supported transparently:
    - sklearn-like estimators (one per output): XGBRegressor, etc.
    - Benjamin's dict payloads (one per output): {model, features, encoders, ...}

Output shape:
  {"<output_key>": {"value": float, "low": float, "high": float}, ...}
"""

from __future__ import annotations

import pandas as pd

from . import benjamin
from .schema import FEATURE_KEYS, OUTPUT_KEYS, REL_WIDTH


def _row(features: dict) -> pd.DataFrame:
    return pd.DataFrame([[float(features[k]) for k in FEATURE_KEYS]], columns=FEATURE_KEYS)


def predict(models: dict, features: dict, lipid: str | None = None) -> dict:
    if any(benjamin.is_payload(v) for v in models.values()):
        return benjamin.predict(models, features, lipid or benjamin.DEFAULT_LIPID)

    row = _row(features)
    out: dict = {}
    for key in OUTPUT_KEYS:
        v = float(models[key].predict(row)[0])
        w = abs(v) * REL_WIDTH[key] + 1e-6
        out[key] = {"value": v, "low": v - w, "high": v + w}
    return out
