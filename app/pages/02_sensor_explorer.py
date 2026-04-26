"""
02_sensor_explorer.py
---------------------
Streamlit page: Interactive sensor trend viewer.  Lets the user select
an engine and a subset of sensors, and displays a subplot grid with
warning-zone annotations.
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
st.title(f"📈 Sensor Explorer — {dataset_choice}")


# ------------------------------------------------------------------ #
# Cached data loader                                                   #
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(name: str):
    """Load and cache the processed train/test DataFrames."""
    return data_processing.prepare_dataset(name)


train_df, _ = load_dataset(dataset_choice)

# ------------------------------------------------------------------ #
# Controls                                                             #
# ------------------------------------------------------------------ #
engine_ids = sorted(train_df["engine_id"].unique())
engine_id = st.slider(
    "Select Engine ID",
    min_value=int(engine_ids[0]),
    max_value=int(engine_ids[-1]),
    value=int(engine_ids[0]),
)

sensor_cols = data_processing.get_sensor_columns(train_df)
selected_sensors = st.multiselect(
    "Select Sensors to Display",
    options=sensor_cols,
    default=sensor_cols,
)

if not selected_sensors:
    st.warning("Please select at least one sensor.")
    st.stop()

# ------------------------------------------------------------------ #
# Sensor grid                                                          #
# ------------------------------------------------------------------ #
fig = utils.plot_sensor_grid_plotly(train_df, engine_id, selected_sensors, dataset_choice)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Engine data table                                                    #
# ------------------------------------------------------------------ #
st.subheader(f"Engine {engine_id} — Raw Data")
df_eng = train_df[train_df["engine_id"] == engine_id]
st.dataframe(df_eng, use_container_width=True)

st.info(
    "🔴 The **vertical red dashed line** marks the cycle at which "
    "RUL ≤ 15 — this is the **Warning Zone**.  Any detection method "
    "should ideally trigger an alert *before* this point."
)
