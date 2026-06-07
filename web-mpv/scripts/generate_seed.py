"""Generate data/seed/example_lab.csv — synthetic but physics-flavored.

Run once to (re)create the seed dataset committed in the repo:
    uv run python scripts/generate_seed.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "seed" / "example_lab.csv"

N = 60
RNG = np.random.default_rng(7)


def main() -> None:
    ion = RNG.uniform(35, 55, N)
    chol = RNG.uniform(30, 50, N)
    helper = RNG.uniform(7, 15, N)
    peg = RNG.uniform(0.8, 3.5, N)
    npr = RNG.uniform(3.0, 9.0, N)
    flow = RNG.uniform(1.5, 4.5, N)
    ph = RNG.uniform(3.5, 6.5, N)

    size = 140 - 6.5 * peg + 0.4 * (ion - 46) + RNG.normal(0, 4, N)
    pdi = 0.08 + 0.015 * np.abs(peg - 1.5) + 0.01 * np.abs(npr - 6) + RNG.normal(0, 0.01, N)
    enc = 92 - 5 * np.abs(ph - 4.0) - 0.8 * np.abs(npr - 6) + RNG.normal(0, 2, N)
    zeta = 2.0 + 0.5 * (ion - 46) + RNG.normal(0, 1.5, N)

    df = pd.DataFrame({
        "ionizable_lipid_pct": ion,
        "cholesterol_pct": chol,
        "helper_lipid_pct": helper,
        "peg_lipid_pct": peg,
        "np_ratio": npr,
        "flow_rate_ratio": flow,
        "ph_buffer": ph,
        "size_nm": np.clip(size, 40, 250),
        "pdi": np.clip(pdi, 0.02, 0.6),
        "encaps_pct": np.clip(enc, 0, 100),
        "zeta_mv": zeta,
    }).round(3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"wrote {len(df)} rows -> {OUT}")


if __name__ == "__main__":
    main()
