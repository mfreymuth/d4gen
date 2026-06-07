"""My Data: manage multiple datasets in-browser (inline Excel I/O + edit + select)."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from lib.datasets import (
    create_dataset,
    delete_dataset,
    empty_dataset,
    list_datasets,
    load_dataset,
    next_default_name,
    save_dataset,
)
from lib.schema import FEATURE_KEYS, OUTPUT_KEYS

SELECT_COL = "✓"


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept `observed_<output>` aliases. Drop extra columns silently."""
    rename = {}
    for k in OUTPUT_KEYS:
        if f"observed_{k}" in df.columns and k not in df.columns:
            rename[f"observed_{k}"] = k
    df = df.rename(columns=rename)
    keep = [c for c in FEATURE_KEYS + OUTPUT_KEYS if c in df.columns]
    return df[keep]


def _to_xlsx(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="data")
    return buf.getvalue()


def _handle_excel(uploaded, dataset_id: str) -> None:
    try:
        new_df = pd.read_excel(io.BytesIO(uploaded.read()))
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    new_df = _normalize_columns(new_df)
    missing = [c for c in FEATURE_KEYS + OUTPUT_KEYS if c not in new_df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Cannot import.")
        return

    existing = load_dataset(dataset_id)
    combined = pd.concat([existing, new_df], ignore_index=True)
    save_dataset(dataset_id, combined)
    st.success(f"Appended {len(new_df)} rows to `{dataset_id}`.")
    st.session_state.pop(f"data_excel_{dataset_id}", None)
    st.rerun()


def _resolve_dataset() -> str | None:
    """Pick / create / rename datasets. Returns the active id."""
    datasets = list_datasets()
    ids = [d["id"] for d in datasets]

    if not ids:
        new_id = next_default_name(ids)
        create_dataset(new_id)
        st.session_state["data_current"] = new_id
        st.rerun()
        return None

    current = st.session_state.get("data_current")
    if current not in ids:
        current = ids[0]
    default_idx = ids.index(current)

    col_pick, col_new, col_del = st.columns(
        [4.4, 1, 1.2], vertical_alignment="bottom", gap="small"
    )
    version = st.session_state.get("data_picker_version", 0)
    with col_pick:
        choice = st.selectbox(
            "Dataset",
            ids,
            index=default_idx,
            key=f"data_picker_v{version}",
        )
    with col_new:
        if st.button("+ New", key="data_new_btn", width="stretch"):
            new_id = next_default_name(ids)
            create_dataset(new_id)
            st.session_state["data_current"] = new_id
            st.session_state["data_picker_version"] = version + 1
            st.rerun()
    with col_del:
        if st.button("Delete dataset", key=f"data_del_ds_{choice}", width="stretch"):
            old_ids = list(ids)
            idx = old_ids.index(choice)
            delete_dataset(choice)
            remaining = [d["id"] for d in list_datasets()]
            next_id = None
            for i in range(idx + 1, len(old_ids)):
                if old_ids[i] in remaining:
                    next_id = old_ids[i]
                    break
            if next_id is None:
                for i in range(idx - 1, -1, -1):
                    if old_ids[i] in remaining:
                        next_id = old_ids[i]
                        break
            if next_id is None and remaining:
                next_id = remaining[0]
            if next_id:
                st.session_state["data_current"] = next_id
            else:
                st.session_state.pop("data_current", None)
            st.session_state["data_picker_version"] = version + 1
            st.success(f"Deleted `{choice}`.")
            st.rerun()

    st.session_state["data_current"] = choice
    return choice


def render() -> None:
    with st.container(border=True):
        dataset_id = _resolve_dataset()
        if dataset_id is None:
            return

        df = load_dataset(dataset_id)
        if df.empty:
            df = empty_dataset()

        display = df.copy()
        display.insert(0, SELECT_COL, False)

        st.caption(
            f"{len(df)} row(s). Edit cells directly, add a row at the bottom, "
            f"or check the `{SELECT_COL}` column to mark rows for deletion."
        )
        edited = st.data_editor(
            display,
            width="stretch",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                SELECT_COL: st.column_config.CheckboxColumn(
                    SELECT_COL, help="Select rows to delete", default=False, width="small"
                ),
            },
            key=f"data_editor_{dataset_id}",
        )

        selected_count = (
            int(edited[SELECT_COL].fillna(False).sum()) if SELECT_COL in edited else 0
        )
        clean = edited.drop(columns=[SELECT_COL], errors="ignore")

        toolbar = st.container(key="data_toolbar")
        col_up, col_dl, col_del, col_save = toolbar.columns(
            [1, 1, 1.2, 1], vertical_alignment="bottom", gap="small"
        )
        with col_up:
            uploaded = st.file_uploader(
                "Load from Excel",
                type=["xlsx", "xls"],
                key=f"data_excel_{dataset_id}",
                label_visibility="collapsed",
            )
            if uploaded is not None:
                _handle_excel(uploaded, dataset_id)
        with col_dl:
            st.download_button(
                "Export to Excel",
                data=_to_xlsx(clean),
                file_name=f"{dataset_id}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"data_export_{dataset_id}",
                width="stretch",
            )
        with col_del:
            if st.button(
                f"Delete selected ({selected_count})",
                key=f"data_delete_{dataset_id}",
                disabled=selected_count == 0,
                width="stretch",
            ):
                keep = edited[~edited[SELECT_COL].fillna(False)].drop(columns=[SELECT_COL])
                keep = keep.dropna(how="all").reset_index(drop=True)
                save_dataset(dataset_id, keep)
                st.success(f"Deleted {selected_count} row(s).")
                st.rerun()
        with col_save:
            if st.button(
                "Save changes",
                type="primary",
                key=f"data_save_{dataset_id}",
                width="stretch",
            ):
                cleaned = clean.dropna(how="all").reset_index(drop=True)
                save_dataset(dataset_id, cleaned)
                st.success(f"Saved {len(cleaned)} row(s) to `{dataset_id}`.")
                st.rerun()
