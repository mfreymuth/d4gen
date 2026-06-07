"""Reusable view: model picker + sliders + predict outputs."""

from __future__ import annotations

import streamlit as st

from lib import benjamin, registry
from lib.inference import predict
from lib.schema import FEATURES, OUTPUTS


def _model_picker(scope: str) -> dict | None:
    models = registry.list_models(scope)
    if not models:
        return None
    labels = [f"{m['name']} · {m['date'][:10]}" for m in models]
    idx = st.selectbox(
        "Model",
        range(len(models)),
        format_func=lambda i: labels[i],
        key=f"{scope}_picker",
    )
    return models[idx]


def metadata_card(meta: dict) -> None:
    r2 = meta.get("r2", {})
    mae = meta.get("mae", {})
    cols = st.columns(len(OUTPUTS))
    for col, (key, label, unit, _dir) in zip(cols, OUTPUTS):
        unit_suffix = f" {unit}" if unit else ""
        col.markdown(f"**{label}**")
        col.metric("R²", f"{r2.get(key, 0):.3f}", label_visibility="visible")
        col.metric("MAE", f"{mae.get(key, 0):.2f}{unit_suffix}", label_visibility="visible")
    n_train = meta.get("n_train") or "?"
    st.caption(
        f"**Name** `{meta.get('name', '?')}` · "
        f"**Trained on** {n_train} rows · "
        f"**Dataset** `{meta.get('dataset_id', '?')}` · "
        f"**Date** {meta.get('date', 'n/a')[:10]}"
    )


def _sliders(scope: str) -> dict:
    cols = st.columns(2)
    values: dict = {}
    for i, (key, label, lo, hi, default, unit) in enumerate(FEATURES):
        col = cols[i % 2]
        suffix = f" ({unit})" if unit else ""
        if key.endswith("_pct"):
            values[key] = col.number_input(
                f"{label}{suffix}",
                min_value=0.0, max_value=100.0,
                value=float(default), step=0.1,
                key=f"{scope}_input_{key}",
            )
        else:
            values[key] = col.slider(
                f"{label}{suffix}",
                float(lo), float(hi), float(default),
                step=(hi - lo) / 100.0,
                key=f"{scope}_slider_{key}",
            )

    composition_total = sum(
        values[k] for k in ("ionizable_lipid_pct", "cholesterol_pct",
                            "helper_lipid_pct", "peg_lipid_pct")
        if k in values
    )
    if abs(composition_total - 100.0) > 0.05:
        st.warning(
            f"Composition total = {composition_total:.1f}% (expected 100%). "
            "The model expects mol fractions that sum to 100%."
        )
    else:
        st.caption(f"Composition total: {composition_total:.1f} %")
    return values


def _outputs(pred: dict) -> None:
    cols = st.columns(len(OUTPUTS))
    for col, (key, label, unit, _) in zip(cols, OUTPUTS):
        v = pred[key]["value"]
        col.metric(label, f"{v:.2f} {unit}".strip())


def render(scope: str, empty_message: str) -> dict | None:
    """Render the predict view for a given scope. Returns selected meta or None."""
    meta = _model_picker(scope)
    if meta is None:
        st.info(empty_message)
        return None

    with st.container(border=True):
        st.markdown("##### Model")
        metadata_card(meta)

    with st.container(border=True):
        st.markdown("##### Formulation inputs")
        values = _sliders(scope)
        lipid = None
        if meta.get("type") == "benjamin":
            c1, c2, c3 = st.columns(3)
            lipid = c1.selectbox(
                "Ionizable lipid",
                benjamin.LIPIDS,
                index=0,
                key=f"{scope}_benjamin_lipid",
                help="Only the two lipids seen during training are supported.",
            )
            values["tfr"] = c2.slider(
                "Total flow rate (mL/min)",
                1.0, 25.0, float(benjamin.DEFAULTS["TFR"]),
                step=0.1,
                key=f"{scope}_benjamin_tfr",
            )
            values["buffer_mm"] = c3.slider(
                "Buffer concentration (mM)",
                5.0, 200.0, float(benjamin.DEFAULTS["Buffer_mM"]),
                step=1.0,
                key=f"{scope}_benjamin_buffer_mm",
            )

    models = registry.load_model(scope, meta["id"])
    pred = predict(models, values, lipid=lipid)

    with st.container(border=True):
        st.markdown("##### Predicted outputs")
        _outputs(pred)

    return meta
