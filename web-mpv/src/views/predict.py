"""Reusable view: model picker + sliders + predict outputs."""

from __future__ import annotations

import time

import streamlit as st

from lib import benjamin, registry
from lib.inference import predict
from lib.schema import FEATURES, OUTPUT_TARGETS, OUTPUTS


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
        lo, hi = OUTPUT_TARGETS[key]
        ok = lo <= v <= hi
        border = "#16A34A" if ok else "#DC2626"
        bg = "#F0FDF4" if ok else "#FEF2F2"
        unit_html = (
            f' <span style="font-size:1rem;color:rgba(0,0,0,.55);font-weight:500;">{unit}</span>'
            if unit else ""
        )
        col.markdown(
            f'<div class="predicted-output-card" style="'
            f'border:2px solid {border};border-radius:8px;'
            f'padding:14px 16px;background:{bg};margin-bottom:0.5rem;">'
            f'<div style="color:rgba(0,0,0,.55);font-size:.8125rem;'
            f'line-height:1.2;margin-bottom:6px;">{label}</div>'
            f'<div style="font-size:1.75rem;font-weight:600;line-height:1;'
            f'color:#0D1B2A;">{v:.2f}{unit_html}</div>'
            f'<div style="color:rgba(0,0,0,.45);font-size:.75rem;'
            f'margin-top:6px;">target: {lo:g} – {hi:g}{(" " + unit) if unit else ""}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


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

    key_card = f"{scope}_output_card"
    key_pred = f"{scope}_last_pred"

    with st.container(border=True, key=key_card):
        title_col, status_col, _ = st.columns(
            [2, 3, 7], vertical_alignment="center", gap="small"
        )
        title_col.markdown("##### Predicted outputs")
        out_slot = st.empty()

        last = st.session_state.get(key_pred)
        if last:
            with out_slot.container():
                _outputs(last)

        with status_col, st.spinner("Computing predictions…"):
            models = _load_model_cached(scope, meta["id"])
            pred = predict(models, values, lipid=lipid)
            time.sleep(1.0)

        with out_slot.container():
            _outputs(pred)
        st.session_state[key_pred] = pred

    return meta


@st.cache_resource(show_spinner=False)
def _load_model_cached(scope: str, model_id: str) -> dict:
    return registry.load_model(scope, model_id)
