"""
03_ewma_analysis.py
-------------------
Streamlit page: Interactive EWMA control-chart analysis.  Provides
sidebar sliders for λ and init_window, sensor selector, and engine
selector.  Displays the Plotly EWMA chart plus key metrics.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from src import data_processing, spc

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
dataset_choice = st.session_state.get("dataset_choice", "FD001")
st.title(f"📉 EWMA Analysis — {dataset_choice}")


# ------------------------------------------------------------------ #
# Cached data loader                                                   #
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(name: str):
    """Load and cache the processed train/test DataFrames."""
    return data_processing.prepare_dataset(name)


train_df, test_df = load_dataset(dataset_choice)

# ------------------------------------------------------------------ #
# Sidebar controls                                                     #
# ------------------------------------------------------------------ #
sensor_cols = data_processing.get_sensor_columns(train_df)

sensor_col = st.sidebar.selectbox(
    "Sensor Column",
    options=sensor_cols,
    index=sensor_cols.index("sensor_12") if "sensor_12" in sensor_cols else 0,
)

lambda_val = st.sidebar.slider(
    "λ (smoothing parameter)",
    min_value=0.05,
    max_value=0.30,
    value=0.10,
    step=0.05,
)

init_window = st.sidebar.slider(
    "Init Window (baseline size)",
    min_value=10,
    max_value=50,
    value=20,
    step=5,
)

engine_ids = sorted(test_df["engine_id"].unique())
engine_id = st.sidebar.number_input(
    "Engine ID",
    min_value=int(engine_ids[0]),
    max_value=int(engine_ids[-1]),
    value=int(engine_ids[0]),
    step=1,
)

# ------------------------------------------------------------------ #
# Run EWMA                                                            #
# ------------------------------------------------------------------ #
df_engine = test_df[test_df["engine_id"] == engine_id].copy()

if df_engine.empty:
    st.error(f"Engine {engine_id} not found in the test set.")
    st.stop()

result = spc.run_ewma_analysis(df_engine, sensor_col, lambda_val, init_window)
fig = spc.plot_ewma_plotly(result, engine_id, dataset_choice)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #
m1, m2, m3, m4 = st.columns(4)
m1.metric("Baseline Mean (μ)", f"{result['mu']:.4f}")
m2.metric("UCL", f"{result['ucl']:.4f}")
m3.metric("LCL", f"{result['lcl']:.4f}")
m4.metric(
    "Breach Cycle",
    str(result["breach_cycle"]) if result["breach_cycle"] is not None else "No Breach Detected",
)

# ------------------------------------------------------------------ #
# FD003-specific warning                                               #
# ------------------------------------------------------------------ #
if dataset_choice == "FD003":
    st.warning(
        "⚠️ **FD003 Note:** This dataset has two fault modes. "
        "If the EWMA chart shows *NO BREACH DETECTED*, this engine "
        "may be failing via **Fault Mode 2 (HPT degradation)**, which "
        f"is not visible in `{sensor_col}`. Switch to the "
        "**Comparison** page to see how ML catches it."
    )

# ------------------------------------------------------------------ #
# Formula explainer                                                    #
# ------------------------------------------------------------------ #
with st.expander("📐 EWMA Formula & Parameters"):
    st.markdown(
        r"""
**EWMA Recursion**

$$Z_t = \lambda\, x_t + (1 - \lambda)\, Z_{t-1}$$

where $Z_0 = \bar{x}_{\text{init}}$ (mean of the first *init_window* observations).

**Control Limits**

$$UCL = \mu + 3\sigma\sqrt{\frac{\lambda}{2 - \lambda}}$$
$$LCL = \mu - 3\sigma\sqrt{\frac{\lambda}{2 - \lambda}}$$

| Parameter | Meaning | Current Value |
|---|---|---|
| λ | Smoothing weight (higher → more reactive) | """
        + f"{lambda_val:.2f}"
        + r""" |
| Init Window | Number of baseline observations | """
        + str(init_window)
        + r""" |
| 3σ | Control-limit width (99.7% coverage) | Fixed |
"""
    )
