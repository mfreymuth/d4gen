"""Backend for Benjamin's pre-trained XGBoost models.

Each `pkl/nanooracle_benjamin_<Output>.pkl` unpickles to a dict:
    {model, features, encoders, lipid_smiles, mae_cv, r2_cv, seuil}

The wrapped `model` is an XGBRegressor expecting 25 features:
    - 9 numeric (composition %, ratios, flow rates, buffer)
    - 3 derived composition ratios
    - 5 label-encoded categoricals (lipid, helper, PEG, cargo, technique)
    - 8 RDKit descriptors of the chosen ionizable lipid

The current UI exposes 7 numeric inputs. We bridge by:
    - Using sensible defaults for TFR, Buffer_mM
    - Letting the user choose the ionizable lipid (MC3 or SM-102, the only
      classes the model was trained on); other categoricals are fixed.
    - Hardcoding RDKit descriptors per lipid (computed once with rdkit).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .schema import OUTPUT_KEYS
from .storage import ROOT

PKL_DIR = ROOT / "pkl"

OUTPUT_FILES: dict[str, str] = {
    "size_nm": "nanooracle_benjamin_Size_nm.pkl",
    "pdi": "nanooracle_benjamin_PDI.pkl",
    "encaps_pct": "nanooracle_benjamin_EEpct.pkl",
    "zeta_mv": "nanooracle_benjamin_Zeta_mV.pkl",
}

FEATURE_ORDER: list[str] = [
    "Ionizable_pct", "Helper_pct", "Chol_pct", "PEG_pct",
    "NP_ratio", "FRR", "TFR", "Buffer_pH", "Buffer_mM",
    "IL_Chol_ratio", "IL_Helper_ratio", "PEG_total_ratio",
    "Lipid (ionizable/cationic)_enc", "Helper Lipid_enc",
    "PEG-Lipid_enc", "Cargo Type_enc", "technique_enc",
    "MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotBonds", "FractionCSP3", "NumAmines",
]

LIPIDS: list[str] = ["MC3", "SM-102"]
DEFAULT_LIPID = "MC3"

DEFAULTS = {"TFR": 8.0, "Buffer_mM": 50.0}

FIXED_CATEGORICALS: list[tuple[str, str]] = [
    ("Helper Lipid", "DSPC"),
    ("PEG-Lipid", "DMG-PEG"),
    ("Cargo Type", "siRNA"),
    ("Technique", "default"),
]

# RDKit descriptors precomputed for the two trainable lipids (computed once
# from their canonical SMILES via Chem.Descriptors / Lipinski).
DESCRIPTORS: dict[str, dict[str, float]] = {
    "MC3": {
        "MolWt": 676.124, "LogP": 12.8582, "TPSA": 55.84,
        "NumHDonors": 0, "NumHAcceptors": 5, "NumRotBonds": 37,
        "FractionCSP3": 0.8605, "NumAmines": 1,
    },
    "SM-102": {
        "MolWt": 796.228, "LogP": 10.8362, "TPSA": 114.76,
        "NumHDonors": 2, "NumHAcceptors": 9, "NumRotBonds": 45,
        "FractionCSP3": 0.8723, "NumAmines": 1,
    },
}

REL_WIDTH = {"size_nm": 0.12, "pdi": 0.25, "encaps_pct": 0.08, "zeta_mv": 0.30}


def is_payload(obj) -> bool:
    return isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "predict")


def all_present() -> bool:
    return PKL_DIR.is_dir() and all((PKL_DIR / f).exists() for f in OUTPUT_FILES.values())


def load_source_payloads() -> dict[str, dict]:
    """Load the four dict payloads directly from `pkl/` using their original filenames."""
    return {key: joblib.load(PKL_DIR / fname) for key, fname in OUTPUT_FILES.items()}


def aggregate_metrics(payloads: dict[str, dict]) -> dict:
    """Extract CV r²/MAE from each payload into the meta.json schema."""
    r2 = {k: float(p["r2_cv"]) for k, p in payloads.items()}
    mae = {k: float(p["mae_cv"]) for k, p in payloads.items()}
    return {
        "r2": r2,
        "mae": mae,
        "r2_mean": sum(r2.values()) / len(r2),
        "mae_mean": sum(mae.values()) / len(mae),
    }


def _encode_lipid(lipid: str, encoders: dict) -> int:
    enc = encoders["Lipid (ionizable/cationic)"]
    classes = list(enc.classes_)
    if lipid not in classes:
        lipid = classes[0]
    return int(classes.index(lipid))


def build_row(features: dict, lipid: str, encoders: dict) -> pd.DataFrame:
    """Construct the 25-feature DataFrame in the model's expected column order."""
    ionizable = float(features["ionizable_lipid_pct"])
    cholesterol = float(features["cholesterol_pct"])
    helper = float(features["helper_lipid_pct"])
    peg = float(features["peg_lipid_pct"])
    total = ionizable + cholesterol + helper + peg

    desc = DESCRIPTORS.get(lipid, DESCRIPTORS[DEFAULT_LIPID])

    row = {
        "Ionizable_pct": ionizable,
        "Helper_pct": helper,
        "Chol_pct": cholesterol,
        "PEG_pct": peg,
        "NP_ratio": float(features["np_ratio"]),
        "FRR": float(features["flow_rate_ratio"]),
        "TFR": float(features.get("tfr", DEFAULTS["TFR"])),
        "Buffer_pH": float(features["ph_buffer"]),
        "Buffer_mM": float(features.get("buffer_mm", DEFAULTS["Buffer_mM"])),
        "IL_Chol_ratio": ionizable / cholesterol if cholesterol else 0.0,
        "IL_Helper_ratio": ionizable / helper if helper else 0.0,
        "PEG_total_ratio": peg / total if total else 0.0,
        "Lipid (ionizable/cationic)_enc": _encode_lipid(lipid, encoders),
        "Helper Lipid_enc": 0,
        "PEG-Lipid_enc": 0,
        "Cargo Type_enc": 0,
        "technique_enc": 0,
        **desc,
    }
    return pd.DataFrame([[row[c] for c in FEATURE_ORDER]], columns=FEATURE_ORDER)


def predict(payloads: dict[str, dict], features: dict, lipid: str = DEFAULT_LIPID) -> dict:
    out: dict = {}
    for key in OUTPUT_KEYS:
        payload = payloads[key]
        row = build_row(features, lipid, payload["encoders"])
        v = float(payload["model"].predict(row)[0])
        w = abs(v) * REL_WIDTH[key] + 1e-6
        out[key] = {"value": v, "low": v - w, "high": v + w}
    return out
