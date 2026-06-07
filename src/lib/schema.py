"""Input/output specs shared across views and backend."""

from __future__ import annotations

FEATURES: list[tuple[str, str, float, float, float, str]] = [
    ("ionizable_lipid_pct", "Ionizable lipid", 20.0, 60.0, 46.3, "mol %"),
    ("cholesterol_pct", "Cholesterol", 20.0, 55.0, 42.7, "mol %"),
    ("helper_lipid_pct", "Helper lipid (DSPC)", 5.0, 20.0, 9.4, "mol %"),
    ("peg_lipid_pct", "PEG-lipid", 0.5, 5.0, 1.6, "mol %"),
    ("np_ratio", "N/P ratio", 2.0, 12.0, 6.0, ""),
    ("flow_rate_ratio", "Flow rate ratio (aq:org)", 1.0, 12.0, 3.0, ""),
    ("ph_buffer", "Buffer pH", 3.0, 8.0, 4.0, ""),
]

OUTPUTS: list[tuple[str, str, str, str]] = [
    ("size_nm", "Particle size", "nm", "band"),
    ("pdi", "PDI", "", "low"),
    ("encaps_pct", "Encapsulation", "%", "high"),
    ("zeta_mv", "Zeta potential", "mV", "band"),
]

FEATURE_KEYS = [f[0] for f in FEATURES]
OUTPUT_KEYS = [o[0] for o in OUTPUTS]
FEATURE_LABELS = {f[0]: f[1] for f in FEATURES}
OUTPUT_LABELS = {o[0]: o[1] for o in OUTPUTS}

REL_WIDTH = {"size_nm": 0.12, "pdi": 0.25, "encaps_pct": 0.08, "zeta_mv": 0.30}
