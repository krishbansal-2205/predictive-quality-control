"""
01_dataset_overview.py
----------------------
Streamlit page: Dataset shapes, summary statistics, and engine
lifetime distribution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import data_processing, utils

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
dataset_choice = st.session_state.get("dataset_choice", "FD001")
st.title(f"📊 Dataset Overview — {dataset_choice}")


# ------------------------------------------------------------------ #
# Cached data loader                                                   #
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(name: str):
    """Load and cache the processed train/test DataFrames."""
    return data_processing.prepare_dataset(name)


train_df, test_df = load_dataset(dataset_choice)

# ------------------------------------------------------------------ #
# Metrics row                                                          #
# ------------------------------------------------------------------ #
col1, col2 = st.columns(2)
col1.metric("Train shape", f"{train_df.shape[0]:,} × {train_df.shape[1]}")
col2.metric("Test shape", f"{test_df.shape[0]:,} × {test_df.shape[1]}")

# ------------------------------------------------------------------ #
# Dataset description                                                  #
# ------------------------------------------------------------------ #
if dataset_choice == "FD001":
    st.info(
        "**FD001** — 100 training engines, 100 test engines.  "
        "1 operating condition, **1 fault mode** (HPC degradation only).  "
        "All engines degrade via the same mechanism, making EWMA a "
        "reliable detection method."
    )
else:
    st.warning(
        "**FD003** — 100 training engines, 100 test engines.  "
        "1 operating condition, **2 fault modes** (HPC *and* HPT degradation).  "
        "Engines failing via Fault Mode 2 (HPT) may not be detectable "
        "by a single-sensor EWMA chart."
    )

# ------------------------------------------------------------------ #
# Data preview                                                         #
# ------------------------------------------------------------------ #
st.subheader("Training Data Preview")
st.dataframe(train_df.head(20), use_container_width=True)

with st.expander("📋 Descriptive Statistics"):
    st.dataframe(train_df.describe(), use_container_width=True)

# ------------------------------------------------------------------ #
# Engine lifetime distribution                                         #
# ------------------------------------------------------------------ #
st.subheader("Engine Lifetime Distribution")
fig = utils.plot_rul_distribution_plotly(train_df, dataset_choice)
st.plotly_chart(fig, use_container_width=True)

lifetimes = train_df.groupby("engine_id")["cycle"].max()
c1, c2, c3 = st.columns(3)
c1.metric("Min Lifetime", f"{lifetimes.min()} cycles")
c2.metric("Mean Lifetime", f"{lifetimes.mean():.0f} cycles")
c3.metric("Max Lifetime", f"{lifetimes.max()} cycles")
