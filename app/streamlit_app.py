"""
streamlit_app.py
----------------
Main Streamlit dashboard entry point.
Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Predictive Quality Control",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Sidebar — Global Controls                                           #
# ------------------------------------------------------------------ #
st.sidebar.title("🏭 Predictive Quality Control")
st.sidebar.markdown("---")

dataset_choice = st.sidebar.selectbox(
    label="Select Dataset",
    options=["FD001", "FD003"],
    help=(
        "FD001: 1 operating condition, 1 fault mode (HPC). "
        "EWMA reliably detects all failures.\n\n"
        "FD003: 1 operating condition, 2 fault modes (HPC + HPT). "
        "EWMA detects Fault Mode 1 only. ML detects both."
    ),
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Dataset:** {dataset_choice}\n\n"
    f"**Fault Modes:** {'1 (HPC only)' if dataset_choice == 'FD001' else '2 (HPC + HPT)'}\n\n"
    f"**Operating Conditions:** 1"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
st.sidebar.markdown(
    "Use the **Pages** menu above to navigate between:\n"
    "- 📊 Dataset Overview\n"
    "- 📈 Sensor Explorer\n"
    "- 📉 EWMA Analysis\n"
    "- 🤖 ML Model\n"
    "- ⚖️ Comparison\n"
    "- 💰 Business Value"
)

# Store dataset choice in session state for all pages to access
st.session_state["dataset_choice"] = dataset_choice

# ------------------------------------------------------------------ #
# Home Page Content                                                    #
# ------------------------------------------------------------------ #
st.title("🏭 Predictive Quality Control Dashboard")
st.subheader(
    "EWMA Control Charts vs. Machine Learning — NASA C-MAPSS Turbofan Engine Dataset"
)

st.markdown(
    """
This dashboard compares two approaches to fault detection in turbofan engines:

| Approach | Method | Strength | Weakness |
|---|---|---|---|
| Traditional SPC | EWMA Control Chart | Interpretable, simple | Single sensor, single fault mode |
| Machine Learning | XGBoost Classifier | Multi-sensor, multi-fault | Less interpretable |

### Why FD001 vs FD003?
- **FD001**: Engines fail from **one cause only** (HPC degradation). EWMA works perfectly here.
- **FD003**: Engines can fail from **two causes** (HPC or HPT degradation). EWMA catches one, misses the other. ML catches both.

Select a dataset from the sidebar and navigate to a page to begin.
"""
)

col1, col2 = st.columns(2)
with col1:
    st.info(
        "**FD001 Selected**\n\n"
        "Best for demonstrating EWMA effectiveness.\n"
        "All engines fail the same way → EWMA is reliable."
        if dataset_choice == "FD001"
        else "**FD001 Available**\n\nSwitch to FD001 in the sidebar to see EWMA working perfectly."
    )
with col2:
    st.warning(
        "**FD003 Selected**\n\n"
        "Best for demonstrating EWMA limitations.\n"
        "Two fault modes → EWMA misses Fault Mode 2 (HPT)."
        if dataset_choice == "FD003"
        else "**FD003 Available**\n\nSwitch to FD003 in the sidebar to see the EWMA blind spot."
    )
