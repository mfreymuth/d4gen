"""NanoOracle — Streamlit entrypoint.

Top horizontal navbar (Vercel-style): logo on the left, tab buttons next to it,
"My account" button on the right. The Streamlit sidebar is hidden entirely.

Sections:
  - Shared Models: read-only models trained on aggregated data.
  - My Models:     private models (per researcher/lab) + training UI.
  - My Data:       manage datasets in-browser (Excel upload + inline edit).
"""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

from sklearn.exceptions import InconsistentVersionWarning

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

import streamlit as st

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lib import benjamin
from lib.storage import DATASETS_DIR, SEED_DIR, SHARED_DIR, ensure_dirs, write_json
from views import data as data_view
from views import predict as predict_view
from views import train as train_view

LOGO_BLACK = SRC / "logo_black.png"
LOGO_SVG = SRC / "logo.svg"
LOGO_PNG = SRC / "logo.png"
FAVICON = SRC / "favicon_512.png"

GLOBAL_CSS = """
<style>
[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"] {
    display: none !important;
}
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: none !important;
    margin-left: 0 !important;
}
.nav-logo {
    display: flex;
    align-items: center;
    height: 42px;
    margin: 0;
    padding: 0;
    overflow: visible;
    position: relative;
}
.nav-logo svg,
.nav-logo img {
    height: 88px;
    width: auto;
    display: block;
    position: absolute;
    top: 50%;
    left: 0;
    transform: translateY(calc(-50% - 8px));
}
.navbar-spacer {
    margin-bottom: 1.5rem;
}
.st-key-nav_shared,
.st-key-nav_mine,
.st-key-nav_data,
.st-key-nav_account {
    display: flex;
    align-items: center;
}
.st-key-nav_shared button,
.st-key-nav_mine button,
.st-key-nav_data button {
    height: 42px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: none !important;
    background-color: transparent !important;
    color: #1A1A1A !important;
    padding: 0 18px !important;
    width: 100% !important;
    min-width: 0 !important;
    white-space: nowrap !important;
}
.st-key-nav_shared button p,
.st-key-nav_mine button p,
.st-key-nav_data button p {
    white-space: nowrap !important;
}
.st-key-nav_shared button:hover,
.st-key-nav_mine button:hover,
.st-key-nav_data button:hover {
    background-color: #F4F4F5 !important;
    color: #1A1A1A !important;
}
.st-key-nav_shared button[kind="primary"],
.st-key-nav_mine button[kind="primary"],
.st-key-nav_data button[kind="primary"] {
    background-color: #E8F0FE !important;
    color: #0A5DF0 !important;
    font-weight: 600 !important;
}
.st-key-nav_account button {
    height: 42px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: 1px solid #E5E7EB !important;
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    padding: 0 18px !important;
    width: 100% !important;
    min-width: 0 !important;
    white-space: nowrap !important;
}
.st-key-nav_account button:hover {
    background-color: #F4F4F5 !important;
}
h1, h2, h3 {
    color: #0D1B2A;
}
[data-testid="stMetricValue"] {
    color: #0A5DF0;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
    border-radius: 12px;
}
[data-testid="stMain"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:has(> [data-testid="stVerticalBlockBorderWrapper"]) {
    margin-bottom: 1rem;
}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
    padding: 1.25rem 1.25rem 1.25rem 1.25rem;
    background: #FFFFFF;
}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] h5 {
    margin: 0 0 0.5rem 0;
    color: #0D1B2A;
    font-weight: 600;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stHorizontalBlock"] {
    gap: 0.5rem;
}
.st-key-data_toolbar [data-testid="stButton"] > button,
.st-key-data_toolbar [data-testid="stDownloadButton"] > button {
    height: 42px !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    border: none !important;
    padding: 0 14px !important;
    width: 100% !important;
}
.st-key-data_toolbar [data-testid="stDownloadButton"] > button,
.st-key-data_toolbar [data-testid="stButton"] > button[kind="secondary"] {
    background-color: #F4F4F5 !important;
    color: #1A1A1A !important;
}
.st-key-data_toolbar [data-testid="stDownloadButton"] > button:hover,
.st-key-data_toolbar [data-testid="stButton"] > button[kind="secondary"]:hover {
    background-color: #E5E7EB !important;
}
.st-key-data_toolbar [data-testid="stButton"] > button[kind="primary"] {
    background-color: #0A5DF0 !important;
    color: #FFFFFF !important;
}
.st-key-data_toolbar [data-testid="stFileUploader"] section,
.st-key-data_toolbar [data-testid="stFileUploaderDropzone"] {
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    min-height: 0 !important;
}
.st-key-data_toolbar [data-testid="stFileUploaderDropzoneInstructions"],
.st-key-data_toolbar [data-testid="stFileUploader"] small {
    display: none !important;
}
.st-key-data_toolbar [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
    width: 100% !important;
    height: 42px !important;
    border-radius: 8px !important;
    background-color: #F4F4F5 !important;
    color: transparent !important;
    border: none !important;
    padding: 0 !important;
    position: relative;
    font-size: 0 !important;
}
.st-key-data_toolbar [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::after {
    content: "Load from Excel";
    color: #1A1A1A;
    font-weight: 500;
    font-size: 1rem;
    font-family: inherit;
    position: absolute;
    left: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    white-space: nowrap;
}
.st-key-data_toolbar [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]:hover {
    background-color: #E5E7EB !important;
}
.st-key-data_picker input,
.st-key-data_picker [data-baseweb="select"] input {
    caret-color: transparent !important;
    pointer-events: none !important;
    user-select: none !important;
}
.st-key-data_picker [data-baseweb="select"] {
    cursor: pointer !important;
}
</style>
"""

SECTIONS = [
    ("Shared Models", "nav_shared"),
    ("My Models", "nav_mine"),
    ("My Data", "nav_data"),
]


@st.cache_data
def _logo_html() -> str:
    """Return HTML for the navbar logo. Prefer logo_black.png, fall back to logo.svg/logo.png."""
    if LOGO_BLACK.exists():
        import base64
        b64 = base64.b64encode(LOGO_BLACK.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="NanoOracle"/>'
    if LOGO_SVG.exists():
        return LOGO_SVG.read_text()
    if LOGO_PNG.exists():
        import base64
        b64 = base64.b64encode(LOGO_PNG.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" alt="NanoOracle"/>'
    return ""


@st.cache_resource
def _bootstrap() -> None:
    ensure_dirs()
    for seed in SEED_DIR.glob("*.csv"):
        target = DATASETS_DIR / seed.name
        if not target.exists():
            shutil.copy2(seed, target)

    _register_benjamin_shared()


def _register_benjamin_shared() -> None:
    target = SHARED_DIR / "shared_v1"
    if target.exists() or not benjamin.all_present():
        return
    target.mkdir(parents=True, exist_ok=True)
    total_size = 0
    for output_key, fname in benjamin.OUTPUT_FILES.items():
        dst = target / f"{output_key}.joblib"
        shutil.copy2(benjamin.PKL_DIR / fname, dst)
        total_size += dst.stat().st_size

    payloads = benjamin.load_source_payloads()
    metrics = benjamin.aggregate_metrics(payloads)

    write_json(target / "meta.json", {
        "id": "shared_v1",
        "type": "benjamin",
        "name": "shared_v1",
        "date": "2026-06-06T00:00:00+00:00",
        "dataset_id": "external",
        "size_bytes": total_size,
        "n_train": 0,
        "note": "Pre-trained externally. Metrics shown are 5-fold CV values.",
        **metrics,
    })


def _topbar() -> str:
    st.session_state.setdefault("section", SECTIONS[0][0])
    current = st.session_state["section"]

    col_logo, col_s, col_m, col_d, col_spacer, col_account = st.columns(
        [1.4, 1.2, 1.1, 1.0, 5, 1.4],
        vertical_alignment="center",
        gap="small",
    )

    with col_logo:
        logo_html = _logo_html()
        if logo_html:
            st.markdown(f'<div class="nav-logo">{logo_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 🧬 NanoOracle")

    nav_cols = [(col_s, SECTIONS[0]), (col_m, SECTIONS[1]), (col_d, SECTIONS[2])]
    for col, (label, key) in nav_cols:
        with col:
            is_active = current == label
            if st.button(
                label,
                key=key,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["section"] = label
                st.rerun()

    with col_spacer:
        st.write("")

    with col_account:
        if st.button("My account", key="nav_account"):
            st.toast("Account screen not implemented yet", icon="👤")

    st.markdown('<div class="navbar-spacer"></div>', unsafe_allow_html=True)
    return st.session_state["section"]


def main() -> None:
    st.set_page_config(
        page_title="NanoOracle",
        page_icon=str(FAVICON) if FAVICON.exists() else "🧬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    _bootstrap()

    section = _topbar()

    if section == "Shared Models":
        st.title("Shared model")
        st.caption("Baseline trained on aggregated data across labs.")
        predict_view.render(
            "shared",
            empty_message="No shared model found. Add a seed dataset under data/seed/ and refresh.",
        )

    elif section == "My Models":
        st.title("My models")
        st.caption("Private models trained with your lab's own data.")
        meta = predict_view.render(
            "mine",
            empty_message="No private model yet. Train one below.",
        )
        st.divider()
        with st.expander("Train a new model", expanded=meta is None):
            train_view.render()

    elif section == "My Data":
        st.title("My data")
        st.caption("Manage your datasets: edit in-browser or upload from Excel.")
        data_view.render()


main()
