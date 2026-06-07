"""Train one XGBRegressor per output on a tabular dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
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


def _clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Coerce to numeric, replace ±inf with NaN, drop rows with any NaN label.

    XGBoost handles NaN in features natively (missing branch in trees), but
    chokes on NaN/inf labels — so we must drop those rows.
    """
    X = df[FEATURE_KEYS].apply(pd.to_numeric, errors="coerce").astype(float)
    Y = df[OUTPUT_KEYS].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    Y = Y.replace([np.inf, -np.inf], np.nan)

    valid = Y.notna().all(axis=1)
    dropped = int((~valid).sum())
    X = X[valid].reset_index(drop=True)
    Y = Y[valid].reset_index(drop=True)
    return X, Y, dropped


def train(df: pd.DataFrame) -> tuple[dict[str, XGBRegressor], dict]:
    missing = [c for c in FEATURE_KEYS + OUTPUT_KEYS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset missing columns: {missing}")

    X, Y, dropped = _clean(df)
    if len(X) < 5:
        raise ValueError(
            f"Only {len(X)} clean rows after removing NaN/inf labels "
            f"(dropped {dropped}). Need at least 5 to train."
        )

    n = len(X)
    n_splits = min(5, n)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=7)

    models: dict[str, XGBRegressor] = {}
    r2 = {}
    mae = {}
    for key in OUTPUT_KEYS:
        cv_pred = cross_val_predict(_new_estimator(), X, Y[key], cv=kf)
        r2[key] = float(r2_score(Y[key], cv_pred))
        mae[key] = float(mean_absolute_error(Y[key], cv_pred))
        final = _new_estimator()
        final.fit(X, Y[key])
        models[key] = final

    metrics = {
        "r2": r2,
        "mae": mae,
        "r2_mean": float(sum(r2.values()) / len(r2)),
        "mae_mean": float(sum(mae.values()) / len(mae)),
        "n_train": int(n),
        "cv_folds": int(n_splits),
        "n_dropped": dropped,
    }
    return models, metrics
