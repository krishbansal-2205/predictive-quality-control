"""
spc.py
------
Exponentially Weighted Moving Average (EWMA) control-chart logic for
Statistical Process Control.  All functions are **pure computation** —
no Streamlit calls.  Plotting helpers are provided for both Matplotlib
(CLI / notebook) and Plotly (Streamlit dashboard).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns


# ------------------------------------------------------------------ #
# EWMA core                                                           #
# ------------------------------------------------------------------ #
def calculate_ewma(
    series: pd.Series, lambda_val: float = 0.1, init_window: int = 20
) -> pd.Series:
    """Compute the EWMA statistic for a univariate time series.

    The recursion is:

    .. math::
        Z_t = \\lambda x_t + (1 - \\lambda) Z_{t-1}

    with :math:`Z_0` initialised to the mean of the first
    *init_window* observations.

    Args:
        series: Raw sensor readings (one engine, one sensor).
        lambda_val: Smoothing parameter λ ∈ (0, 1].
        init_window: Number of initial observations used to estimate
            the process mean.

    Returns:
        A :class:`pd.Series` of EWMA values with the same index as
        *series*.
    """
    values = series.values.astype(float)
    z = np.empty_like(values)
    # Initialise Z_0 using the EWMA recursion applied to the first
    # observation, with the baseline mean as the prior.
    mu_init = np.mean(values[:init_window])
    z[0] = lambda_val * values[0] + (1 - lambda_val) * mu_init

    for t in range(1, len(values)):
        z[t] = lambda_val * values[t] + (1 - lambda_val) * z[t - 1]

    return pd.Series(z, index=series.index, name="ewma")


def calculate_control_limits(
    series: pd.Series, lambda_val: float = 0.1, init_window: int = 20
) -> Tuple[float, float, float]:
    """Derive the steady-state 3-σ EWMA control limits.

    .. math::
        UCL = \\mu + 3\\sigma\\sqrt{\\frac{\\lambda}{2 - \\lambda}}

    Args:
        series: Raw sensor readings used for baseline estimation.
        lambda_val: EWMA smoothing parameter.
        init_window: Number of initial observations for μ / σ.

    Returns:
        A ``(mu, ucl, lcl)`` tuple.
    """
    baseline = series.iloc[:init_window]
    mu = float(baseline.mean())
    # Use ddof=0 (population std) — consistent with classical SPC charting.
    sigma = float(baseline.std(ddof=0))
    # Guard: if the baseline is perfectly flat, fall back to a small
    # epsilon so that UCL != LCL and the chart remains meaningful.
    if sigma == 0.0:
        sigma = 1e-6
    factor = 3 * sigma * np.sqrt(lambda_val / (2 - lambda_val))
    ucl = mu + factor
    lcl = mu - factor
    return mu, ucl, lcl


def detect_breach(
    ewma_series: pd.Series,
    cycle_series: pd.Series,
    ucl: float,
    lcl: float,
) -> Optional[int]:
    """Find the first cycle at which the EWMA breaches a control limit.

    Args:
        ewma_series: EWMA values.
        cycle_series: Corresponding cycle numbers.
        ucl: Upper control limit.
        lcl: Lower control limit.

    Returns:
        The cycle number of the first breach, or ``None`` if the
        process never leaves the control band.
    """
    breach_mask = (ewma_series.values > ucl) | (ewma_series.values < lcl)
    if breach_mask.any():
        idx = np.argmax(breach_mask)
        return int(cycle_series.iloc[idx])
    return None


# ------------------------------------------------------------------ #
# Convenience runner                                                   #
# ------------------------------------------------------------------ #
def run_ewma_analysis(
    df_engine: pd.DataFrame,
    sensor_col: str,
    lambda_val: float = 0.1,
    init_window: int = 20,
) -> Dict:
    """Run the full EWMA pipeline for a single engine and sensor.

    Args:
        df_engine: DataFrame filtered to a single engine.
        sensor_col: Name of the sensor column to analyse.
        lambda_val: EWMA smoothing parameter.
        init_window: Baseline window size.

    Returns:
        Dictionary with keys ``ewma_values``, ``mu``, ``ucl``, ``lcl``,
        ``breach_cycle``, ``sensor_col``, and ``cycles``.

    Raises:
        KeyError: If *sensor_col* is not in *df_engine*.
    """
    if sensor_col not in df_engine.columns:
        raise KeyError(f"Column '{sensor_col}' not found in DataFrame.")

    series = df_engine[sensor_col].reset_index(drop=True)
    cycles = df_engine["cycle"].reset_index(drop=True)

    ewma_vals = calculate_ewma(series, lambda_val, init_window)
    mu, ucl, lcl = calculate_control_limits(series, lambda_val, init_window)
    breach = detect_breach(ewma_vals, cycles, ucl, lcl)

    return {
        "ewma_values": ewma_vals,
        "mu": mu,
        "ucl": ucl,
        "lcl": lcl,
        "breach_cycle": breach,
        "sensor_col": sensor_col,
        "cycles": cycles,
        "raw_values": series,
    }


# ------------------------------------------------------------------ #
# Matplotlib plotting (CLI / notebooks)                                #
# ------------------------------------------------------------------ #
def plot_ewma_matplotlib(
    ewma_result: Dict, engine_id: int, save_path: Path
) -> None:
    """Save a static EWMA control chart using Matplotlib + Seaborn.

    Args:
        ewma_result: Dictionary returned by :func:`run_ewma_analysis`.
        engine_id: Engine identifier (used in the title).
        save_path: File path for the saved PNG figure.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(14, 5))

    cycles = ewma_result["cycles"]
    raw = ewma_result["raw_values"]
    ewma = ewma_result["ewma_values"]
    mu = ewma_result["mu"]
    ucl = ewma_result["ucl"]
    lcl = ewma_result["lcl"]
    breach = ewma_result["breach_cycle"]
    sensor = ewma_result["sensor_col"]

    ax.plot(cycles, raw, color="lightgray", linewidth=0.8, label="Raw sensor")
    ax.plot(cycles, ewma, color="royalblue", linewidth=1.5, label="EWMA")
    ax.axhline(ucl, color="red", linestyle="--", linewidth=1, label=f"UCL = {ucl:.4f}")
    ax.axhline(lcl, color="red", linestyle="--", linewidth=1, label=f"LCL = {lcl:.4f}")
    ax.axhline(mu, color="green", linestyle="--", linewidth=1, label=f"μ = {mu:.4f}")

    if breach is not None:
        ax.axvline(breach, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.annotate(
            f"Breach @ cycle {breach}",
            xy=(breach, ucl),
            fontsize=10,
            color="red",
            fontweight="bold",
        )
    else:
        ax.annotate(
            "NO BREACH DETECTED",
            xy=(0.5, 0.95),
            xycoords="axes fraction",
            fontsize=12,
            color="orange",
            fontweight="bold",
            ha="center",
        )

    ax.set_title(f"EWMA Control Chart - Engine {engine_id} | {sensor}", fontsize=13)
    ax.set_xlabel("Cycle")
    ax.set_ylabel(sensor)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Plotly plotting (Streamlit dashboard)                                #
# ------------------------------------------------------------------ #
def plot_ewma_plotly(
    ewma_result: Dict, engine_id: int, dataset_name: str
) -> go.Figure:
    """Build an interactive Plotly EWMA control chart.

    Args:
        ewma_result: Dictionary returned by :func:`run_ewma_analysis`.
        engine_id: Engine identifier.
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        A :class:`plotly.graph_objects.Figure` ready for ``st.plotly_chart``.
    """
    cycles = ewma_result["cycles"]
    raw = ewma_result["raw_values"]
    ewma = ewma_result["ewma_values"]
    mu = ewma_result["mu"]
    ucl = ewma_result["ucl"]
    lcl = ewma_result["lcl"]
    breach = ewma_result["breach_cycle"]
    sensor = ewma_result["sensor_col"]

    fig = go.Figure()

    # Raw sensor
    fig.add_trace(
        go.Scatter(
            x=cycles, y=raw, mode="lines", name="Raw sensor",
            line=dict(color="lightgray", width=1),
        )
    )

    # EWMA
    fig.add_trace(
        go.Scatter(
            x=cycles, y=ewma, mode="lines", name="EWMA",
            line=dict(color="royalblue", width=2.5),
        )
    )

    # Control limits
    fig.add_hline(y=ucl, line_dash="dash", line_color="red",
                  annotation_text=f"UCL = {ucl:.4f}", annotation_position="top left")
    fig.add_hline(y=lcl, line_dash="dash", line_color="red",
                  annotation_text=f"LCL = {lcl:.4f}", annotation_position="bottom left")
    fig.add_hline(y=mu, line_dash="dash", line_color="green",
                  annotation_text=f"μ = {mu:.4f}", annotation_position="top right")

    if breach is not None:
        fig.add_vline(x=breach, line_dash="dash", line_color="red", line_width=2)
        fig.add_annotation(
            x=breach, y=ucl,
            text=f"⚠ Breach @ cycle {breach}",
            showarrow=True, arrowhead=2, font=dict(color="red", size=13),
        )
    else:
        fig.add_annotation(
            x=cycles.iloc[len(cycles) // 2],
            y=ucl,
            text="WARNING: NO BREACH - Possible Fault Mode 2 (HPT)",
            showarrow=False,
            font=dict(color="orange", size=14),
            bgcolor="rgba(255,165,0,0.15)",
            bordercolor="orange",
            borderwidth=1,
        )

    fig.update_layout(
        title=f"EWMA Control Chart - {dataset_name} Engine {engine_id} | {sensor}",
        xaxis_title="Cycle",
        yaxis_title=sensor,
        template="plotly_white",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    return fig
