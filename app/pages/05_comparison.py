"""
05_comparison.py
----------------
Streamlit page: Side-by-side EWMA vs ML comparison.
- FD001 mode: single engine view.
- FD003 mode: dual-engine view to highlight the SPC blind spot.
"""

import sys
from pathlib import Path

# Insert project root BEFORE any src imports so the package is always found.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import data_processing, modeling, spc, utils  # noqa: E402
import streamlit as st
import pandas as pd

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
dataset_choice = st.session_state.get("dataset_choice", "FD001")
st.title(f"⚖️ EWMA vs ML Comparison — {dataset_choice}")


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
# Model availability check                                             #
# ------------------------------------------------------------------ #
if not modeling.model_exists(dataset_choice):
    st.warning(
        f"⚠️ No trained model found for **{dataset_choice}**. "
        "Please go to the **ML Model** page and train one first."
    )
    st.stop()

model = cached_load_model(dataset_choice)
engine_ids = sorted(test_df["engine_id"].unique())

# Compute near-failure engines: min RUL per engine in the test sequence
# equals true_RUL (the RUL at the last recorded cycle). Only engines
# with true_RUL <= 30 can have failure_within_window=1 in the test set.
_min_rul = test_df.groupby("engine_id")["RUL"].min()
near_failure_ids = sorted(_min_rul[_min_rul <= 30].index.tolist())

if near_failure_ids:
    st.info(
        f"💡 **Near-failure engines** (true RUL ≤ 30 cycles at data cutoff — "
        f"select these to see the ML model trigger a warning): "
        f"`{near_failure_ids[:15]}`"
    )


# ------------------------------------------------------------------ #
# Helper: analyse one engine                                           #
# ------------------------------------------------------------------ #
def analyse_engine(eng_id: int, container):
    """Run EWMA + ML analysis and render results in *container*."""
    df_eng = test_df[test_df["engine_id"] == eng_id].copy()
    if df_eng.empty:
        container.error(f"Engine {eng_id} not found.")
        return None

    actual_failure = int(df_eng["cycle"].max() + df_eng["RUL"].iloc[-1])

    # EWMA
    ewma_result = spc.run_ewma_analysis(df_eng, "sensor_12")
    fig_ewma = spc.plot_ewma_plotly(ewma_result, eng_id, dataset_choice)
    container.plotly_chart(fig_ewma, use_container_width=True)

    # ML
    ml_warning, proba_series = modeling.predict_failure_start(
        model, df_eng, threshold=0.3)
    cycle_series = df_eng["cycle"].reset_index(drop=True)
    proba_reset = proba_series.reset_index(drop=True)
    fig_ml = utils.plot_probability_timeline_plotly(
        cycle_series, proba_reset, eng_id, ml_warning, actual_failure, dataset_choice,
    )
    container.plotly_chart(fig_ml, use_container_width=True)

    # Metrics
    breach = ewma_result["breach_cycle"]
    mc1, mc2, mc3 = container.columns(3)
    mc1.metric("EWMA Breach", str(breach) if breach else "No Breach")
    mc2.metric("ML Warning", str(ml_warning) if ml_warning else "No Warning")

    ewma_lead = (actual_failure - breach) if breach else 0
    ml_lead = (actual_failure - ml_warning) if ml_warning else 0
    advantage = ml_lead - ewma_lead
    mc3.metric("ML Lead Advantage", f"{advantage} cycles")

    # Alert boxes
    if breach is None:
        container.error(
            f"❌ EWMA failed to detect failure for Engine {eng_id}. "
            "This is the SPC blind spot in FD003."
        )
    if ml_warning is not None:
        container.success(
            f"✅ ML model successfully predicted failure for Engine {eng_id} "
            f"at cycle {ml_warning}."
        )
    elif ml_warning is None:
        container.warning(
            f"⚠️ ML model did not predict failure for Engine {eng_id} "
            "above the 0.3 threshold."
        )

    return {
        "Engine ID": eng_id,
        "EWMA Breach": breach if breach else "N/A",
        "ML Warning": ml_warning if ml_warning else "N/A",
        "Actual Failure": actual_failure,
        "EWMA Lead Time": ewma_lead,
        "ML Lead Time": ml_lead,
    }


# ------------------------------------------------------------------ #
# FD001 mode: single engine                                            #
# ------------------------------------------------------------------ #
if dataset_choice == "FD001":
    _default_idx = (
        engine_ids.index(near_failure_ids[0]) if near_failure_ids else 0
    )
    engine_id = st.selectbox("Select Engine", engine_ids, index=_default_idx)
    analyse_engine(engine_id, st)

# ------------------------------------------------------------------ #
# FD003 mode: dual-engine comparison                                   #
# ------------------------------------------------------------------ #
else:
    col_left, col_right = st.columns(2)
    _idx1 = engine_ids.index(near_failure_ids[0]) if near_failure_ids else 0
    _idx2 = (
        engine_ids.index(near_failure_ids[1])
        if len(near_failure_ids) > 1
        else min(10, len(engine_ids) - 1)
    )
    with col_left:
        eng1 = st.selectbox("Engine 1", engine_ids, index=_idx1, key="eng1")
    with col_right:
        eng2 = st.selectbox(
            "Engine 2", engine_ids,
            index=_idx2,
            key="eng2",
        )

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader(f"Engine {eng1}")
        r1 = analyse_engine(eng1, left)
    with right:
        st.subheader(f"Engine {eng2}")
        r2 = analyse_engine(eng2, right)

    st.markdown("---")
    st.subheader("Comparison Table")
    rows = [r for r in (r1, r2) if r is not None]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
