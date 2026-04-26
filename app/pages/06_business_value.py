"""
06_business_value.py
--------------------
Streamlit page: Cost-benefit analysis comparing EWMA vs ML detection
for a selected engine.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

from src import data_processing, modeling, spc, utils

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
dataset_choice = st.session_state.get("dataset_choice", "FD001")
st.title(f"💰 Business Value — {dataset_choice}")


# ------------------------------------------------------------------ #
# Cached helpers                                                       #
# ------------------------------------------------------------------ #
@st.cache_data(show_spinner="Loading dataset…")
def load_dataset(name: str):
    """Load and cache the processed train/test DataFrames."""
    return data_processing.prepare_dataset(name)


@st.cache_resource(show_spinner="Loading model…")
def cached_load_model(name: str):
    """Load a persisted model (cached as a resource)."""
    return modeling.load_model(name)


train_df, test_df = load_dataset(dataset_choice)

# ------------------------------------------------------------------ #
# Model check                                                         #
# ------------------------------------------------------------------ #
if not modeling.model_exists(dataset_choice):
    st.warning(
        f"⚠️ No trained model for **{dataset_choice}**. "
        "Go to the **ML Model** page to train first."
    )
    st.stop()

model = cached_load_model(dataset_choice)

# ------------------------------------------------------------------ #
# User controls                                                        #
# ------------------------------------------------------------------ #
engine_ids = sorted(test_df["engine_id"].unique())

engine_id = st.number_input(
    "Engine ID",
    min_value=int(engine_ids[0]),
    max_value=int(engine_ids[-1]),
    value=int(engine_ids[0]),
    step=1,
)

c1, c2, c3 = st.columns(3)
with c1:
    cost_defect = st.number_input(
        "Cost of Undetected Defect ($)",
        min_value=0,
        value=50_000,
        step=5_000,
    )
with c2:
    cost_false_alarm = st.number_input(
        "Cost of False Alarm ($)",
        min_value=0,
        value=2_000,
        step=500,
    )
with c3:
    threshold = st.slider(
        "ML Warning Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

# ------------------------------------------------------------------ #
# Analysis                                                             #
# ------------------------------------------------------------------ #
df_eng = test_df[test_df["engine_id"] == engine_id].copy()

if df_eng.empty:
    st.error(f"Engine {engine_id} not found.")
    st.stop()

actual_failure = int(df_eng["cycle"].max() + df_eng["RUL"].iloc[-1])

# EWMA
ewma_result = spc.run_ewma_analysis(df_eng, "sensor_12")
ewma_breach = ewma_result["breach_cycle"]

# ML
ml_warning, _ = modeling.predict_failure_start(model, df_eng, threshold=threshold)

# Business value
bv = utils.calculate_business_value(
    engine_id, dataset_choice, actual_failure,
    ewma_breach, ml_warning,
    cost_defect=cost_defect, cost_false_alarm=cost_false_alarm,
)

# ------------------------------------------------------------------ #
# Lead-time metrics                                                    #
# ------------------------------------------------------------------ #
st.subheader("Lead Time")
lt1, lt2, lt3 = st.columns(3)
lt1.metric("EWMA Lead Time", f"{bv['ewma_lead_time']} cycles")
lt2.metric("ML Lead Time", f"{bv['ml_lead_time']} cycles")
lt3.metric("ML Advantage", f"{bv['ml_lead_time'] - bv['ewma_lead_time']} cycles")

# ------------------------------------------------------------------ #
# Financial metrics                                                    #
# ------------------------------------------------------------------ #
st.subheader("Financial Impact")
f1, f2, f3 = st.columns(3)
f1.metric("EWMA Value Saved", f"${bv['ewma_value_saved']:,.0f}")
f2.metric("ML Value Saved", f"${bv['ml_value_saved']:,.0f}")
f3.metric("ML Financial Advantage", f"${bv['ml_advantage']:,.0f}")

# ------------------------------------------------------------------ #
# Formatted report                                                     #
# ------------------------------------------------------------------ #
st.subheader("Detailed Report")
report_str = utils.format_business_value_report(bv)
st.code(report_str, language="text")

# ------------------------------------------------------------------ #
# Bar chart                                                            #
# ------------------------------------------------------------------ #
st.subheader("EWMA vs ML — Value Saved")
fig = go.Figure(
    data=[
        go.Bar(
            name="EWMA",
            x=["Value Saved"],
            y=[bv["ewma_value_saved"]],
            marker_color="steelblue",
        ),
        go.Bar(
            name="ML (XGBoost)",
            x=["Value Saved"],
            y=[bv["ml_value_saved"]],
            marker_color="darkorange",
        ),
    ]
)
fig.update_layout(
    barmode="group",
    title=f"Cost Avoidance — {dataset_choice} Engine {engine_id}",
    yaxis_title="Dollars ($)",
    template="plotly_white",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# FD003 note                                                           #
# ------------------------------------------------------------------ #
if dataset_choice == "FD003":
    st.info(
        "**FD003 Note:** This dataset contains two fault modes. "
        "For engines failing via Fault Mode 2 (HPT degradation), "
        "the EWMA value saved will be **$0** because the breach is "
        "never detected by a single-sensor control chart. "
        "The ML model, using all sensors jointly, can still detect "
        "these failures — demonstrating the financial advantage of ML."
    )
