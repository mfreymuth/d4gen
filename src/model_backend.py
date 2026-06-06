"""
NanoOracle — model backend.

THIS FILE IS THE CONTRACT WITH THE DATA SCIENTISTS.
The front (app.py) only ever calls predict() and the FEATURES/OUTPUTS specs.
Right now everything is mocked. To go live, the DS team replaces the body of
predict() (and optionally load_model()) — the signatures must NOT change.
"""

from __future__ import annotations

import numpy as np

# ---- Input feature spec ---------------------------------------------------
# (name, label, min, max, default, unit). Edit to match the real model inputs.
FEATURES = [
    ("ionizable_lipid_pct", "Ionizable lipid", 20.0, 60.0, 46.3, "mol %"),
    ("cholesterol_pct", "Cholesterol", 20.0, 55.0, 42.7, "mol %"),
    ("helper_lipid_pct", "Helper lipid (DSPC)", 5.0, 20.0, 9.4, "mol %"),
    ("peg_lipid_pct", "PEG-lipid", 0.5, 5.0, 1.6, "mol %"),
    ("np_ratio", "N/P ratio", 2.0, 12.0, 6.0, ""),
    ("flow_rate_ratio", "Flow rate ratio (aq:org)", 1.0, 5.0, 3.0, ""),
    ("ph_buffer", "Buffer pH", 3.0, 8.0, 4.0, ""),
]

# ---- Output spec ----------------------------------------------------------
# (key, label, unit, good_direction)  good_direction in {"low","high","band"}
OUTPUTS = [
    ("size_nm", "Particle size", "nm", "band"),  # ~80-120 nm sweet spot
    ("pdi", "PDI", "", "low"),  # < 0.2 desirable
    ("encaps_pct", "Encapsulation", "%", "high"),  # > 80% desirable
    ("zeta_mv", "Zeta potential", "mV", "band"),
]

_RNG = np.random.default_rng(7)


def load_model(path: str | None = None):
    """DS team: load and return your trained estimator here. Mock returns None."""
    return None


def _mock_point(features: dict) -> dict:
    """Plausible physics-flavored mock so the demo looks credible, not random."""
    ion = features["ionizable_lipid_pct"]
    peg = features["peg_lipid_pct"]
    npr = features["np_ratio"]
    ph = features["ph_buffer"]

    # PEG drives size down; low pH (protonation) and good N/P drive encapsulation up.
    size = 140 - 6.5 * peg + 0.4 * (ion - 46) + _RNG.normal(0, 4)
    pdi = 0.08 + 0.015 * abs(peg - 1.5) + 0.01 * abs(npr - 6) + _RNG.normal(0, 0.01)
    enc = 92 - 5 * abs(ph - 4.0) - 0.8 * abs(npr - 6) + _RNG.normal(0, 2)
    zeta = 2.0 + 0.5 * (ion - 46) + _RNG.normal(0, 1.5)

    return {
        "size_nm": float(np.clip(size, 40, 250)),
        "pdi": float(np.clip(pdi, 0.02, 0.6)),
        "encaps_pct": float(np.clip(enc, 0, 100)),
        "zeta_mv": float(zeta),
    }


def predict(features: dict) -> dict:
    """
    Core contract. Input: dict mapping every FEATURES key -> float.
    Output: dict mapping every OUTPUTS key -> {"value", "low", "high"}
            plus "shap": {feature_key: contribution_float} for the size output.

    DS team: return real predictions + intervals (quantile reg / conformal)
    + SHAP values. Keep this exact shape.
    """
    point = _mock_point(features)

    # Mock 90% intervals: wider because the dataset is tiny — and that's the point.
    out = {}
    rel_width = {"size_nm": 0.12, "pdi": 0.25, "encaps_pct": 0.08, "zeta_mv": 0.30}
    for key, _, _, _ in OUTPUTS:
        v = point[key]
        w = abs(v) * rel_width[key] + 1e-6
        out[key] = {"value": v, "low": v - w, "high": v + w}

    # Mock SHAP contributions for the size prediction (sum ≈ deviation from base).
    base = 110.0
    contribs = {
        "peg_lipid_pct": -6.5 * (features["peg_lipid_pct"] - 1.5),
        "ionizable_lipid_pct": 0.4 * (features["ionizable_lipid_pct"] - 46),
        "np_ratio": 1.2 * (features["np_ratio"] - 6),
        "flow_rate_ratio": -2.0 * (features["flow_rate_ratio"] - 3),
        "cholesterol_pct": 0.3 * (features["cholesterol_pct"] - 42),
        "helper_lipid_pct": 0.5 * (features["helper_lipid_pct"] - 9),
        "ph_buffer": 0.8 * (features["ph_buffer"] - 4),
    }
    out["shap"] = {"base": base, "contribs": contribs}
    return out
