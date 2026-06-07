"""Train a new private model from a selected dataset."""

from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from lib import registry
from lib.datasets import list_datasets, load_dataset
from lib.training import train
from views import predict as predict_view


def render() -> None:
    datasets = list_datasets()
    if not datasets:
        st.warning("No datasets available. Create one under *My Data* first.")
        return

    with st.container(border=True):
        st.markdown("##### Dataset to train on")
        labels = [f"{d['id']} · {d['n_rows']} rows" for d in datasets]
        idx = st.selectbox(
            "Dataset",
            range(len(datasets)),
            format_func=lambda i: labels[i],
            key="train_dataset_picker",
            label_visibility="collapsed",
        )
        ds = datasets[idx]
        df = load_dataset(ds["id"])
        st.caption(f"Preview (first 5 of {len(df)} rows)")
        st.dataframe(df.head(), width="stretch")

        st.markdown("##### Model name")
        default_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        name = st.text_input(
            "Model name", value=default_name,
            key="train_model_name", label_visibility="collapsed",
        )

        if st.button("Train", type="primary", key="train_btn"):
            with st.spinner("Training…"):
                try:
                    models, metrics = train(df)
                except ValueError as e:
                    st.error(f"Training failed: {e}")
                    return
                model_id = registry.save_model("mine", name, models, metrics, ds["id"])
            dropped = metrics.get("n_dropped", 0)
            if dropped:
                st.warning(
                    f"Dropped {dropped} row(s) with NaN/inf labels before training."
                )
            st.success(f"Trained `{name}` (id `{model_id}`).")
            predict_view.metadata_card({
                "name": name,
                "id": model_id,
                "dataset_id": ds["id"],
                "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "n_train": metrics["n_train"],
                "r2": metrics["r2"],
                "mae": metrics["mae"],
            })
            # Force the picker to land on the newly trained model after rerun.
            # registry.list_models() sorts by date desc, so the new model is at idx 0.
            st.session_state.pop("mine_picker", None)
            st.session_state.pop("mine_last_pred", None)
            st.rerun()
