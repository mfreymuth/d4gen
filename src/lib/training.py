"""Train one XGBRegressor per output on a tabular dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from .schema import FEATURE_KEYS, OUTPUT_KEYS


def _new_estimator() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=7,
        objective="reg:squarederror",
        n_jobs=1,
        verbosity=0,
    )


def train(df: pd.DataFrame) -> tuple[dict[str, XGBRegressor], dict]:
    missing = [c for c in FEATURE_KEYS + OUTPUT_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    X = df[FEATURE_KEYS].astype(float)
    Y = df[OUTPUT_KEYS].astype(float)

    n = len(df)
    test_frac = 0.2 if n >= 25 else max(1 / n, 0.1)
    X_tr, X_te, Y_tr, Y_te = train_test_split(
        X, Y, test_size=test_frac, random_state=7
    )

    models: dict[str, XGBRegressor] = {}
    r2 = {}
    mae = {}
    for key in OUTPUT_KEYS:
        m = _new_estimator()
        m.fit(X_tr, Y_tr[key])
        pred = m.predict(X_te)
        r2[key] = float(r2_score(Y_te[key], pred))
        mae[key] = float(mean_absolute_error(Y_te[key], pred))
        models[key] = m

    metrics = {
        "r2": r2,
        "mae": mae,
        "r2_mean": float(sum(r2.values()) / len(r2)),
        "mae_mean": float(sum(mae.values()) / len(mae)),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
    }
    return models, metrics
