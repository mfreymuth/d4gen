"""
NanoOracle — demo front.

Run:  streamlit run app.py
Three screens built for a 5-min pitch: Predict, Explain, Impact.
All model logic lives in model_backend.py (the DS contract). This file never
touches the model internals — only predict() and the FEATURES/OUTPUTS specs.
"""

import pandas as pd
import streamlit as st

from model_backend import FEATURES, OUTPUTS, predict

st.set_page_config(page_title="NanoOracle", page_icon="🧬", layout="wide")

# ---- sidebar: the formulation the user is designing -----------------------
st.sidebar.title("🧬 NanoOracle")
st.sidebar.caption("Predict LNP physicochemistry before the bench.")
st.sidebar.markdown("### Formulation inputs")

feat_values = {}
for key, label, lo, hi, default, unit in FEATURES:
    suffix = f" ({unit})" if unit else ""
    feat_values[key] = st.sidebar.slider(
        f"{label}{suffix}",
        float(lo),
        float(hi),
        float(default),
        step=(hi - lo) / 100.0,
    )

pred = predict(feat_values)

tab_predict, tab_explain, tab_impact = st.tabs(
    ["🔮 Predict", "🔍 Explain", "📈 Impact"]
)

# ===========================================================================
# 1. PREDICT — the core. Values + confidence intervals.
# ===========================================================================
with tab_predict:
    st.subheader("Predicted outputs")
    st.caption(
        "90% confidence intervals. Wide bands reflect a small dataset — "
        "that gap is exactly the case for pooling failed-experiment data."
    )

    cols = st.columns(len(OUTPUTS))
    for col, (key, label, unit, direction) in zip(cols, OUTPUTS):
        v = pred[key]["value"]
        lo, hi = pred[key]["low"], pred[key]["high"]
        col.metric(label, f"{v:.1f} {unit}".strip())
        col.caption(f"[{lo:.1f} – {hi:.1f}]")

    # quick visual: predicted value with its interval as an error bar
    st.markdown("##### Prediction with uncertainty")
    df = pd.DataFrame(
        {
            "output": [lbl for _, lbl, _, _ in OUTPUTS],
            "value": [pred[k]["value"] for k, _, _, _ in OUTPUTS],
            "low": [pred[k]["low"] for k, _, _, _ in OUTPUTS],
            "high": [pred[k]["high"] for k, _, _, _ in OUTPUTS],
        }
    )
    st.dataframe(
        df.set_index("output").style.format("{:.2f}"), use_container_width=True
    )

# ===========================================================================
# 2. EXPLAIN — SHAP for the size prediction. "Not a black box."
# ===========================================================================
with tab_explain:
    st.subheader("Why this prediction? (SHAP — particle size)")
    st.caption(
        "Each bar = how much a feature pushes size away from the baseline. "
        "Speaks to chemists: the model is interpretable, not a black box."
    )

    shap = pred["shap"]
    contribs = shap["contribs"]
    # order by absolute impact
    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    labels = {k: lbl for k, lbl, *_ in FEATURES}

    chart_df = pd.DataFrame(
        {
            "feature": [labels.get(k, k) for k, _ in items],
            "contribution_nm": [v for _, v in items],
        }
    ).set_index("feature")

    st.bar_chart(chart_df, horizontal=True)
    st.caption(
        f"Baseline size: {shap['base']:.0f} nm  →  "
        f"predicted: {pred['size_nm']['value']:.1f} nm"
    )

# ===========================================================================
# 3. IMPACT — the money slide, made visual. potential/pertinence criterion.
# ===========================================================================
with tab_impact:
    st.subheader("Research time saved")
    st.caption(
        "Adjust the assumptions live — this is a defensible estimate, not a P&L."
    )

    c1, c2, c3 = st.columns(3)
    n_candidates = c1.number_input(
        "Candidate formulations / project", 10, 1000, 100, 10
    )
    weeks_per_cycle = c2.number_input("Weeks per bench cycle", 1, 12, 3, 1)
    filter_rate = c3.slider("% candidates model flags non-viable", 0, 90, 60, 5)

    avoided = int(n_candidates * filter_rate / 100)
    weeks_saved = avoided * weeks_per_cycle
    months_saved = weeks_saved / 4.3

    m1, m2, m3 = st.columns(3)
    m1.metric("Formulations skipped", f"{avoided}")
    m2.metric("Bench weeks saved", f"{weeks_saved}")
    m3.metric("≈ months of research", f"{months_saved:.1f}")

    st.info(
        f"By discarding the **{filter_rate}%** of candidates the model predicts "
        f"non-viable *before* the bench, a project recovers "
        f"**~{months_saved:.0f} months** of wet-lab time."
    )

st.sidebar.divider()
st.sidebar.caption("Mocked backend — swap model_backend.predict() to go live.")
