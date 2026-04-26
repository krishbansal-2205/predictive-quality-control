"""
utils.py
--------
Shared utility functions: directory creation, business-value accounting,
and Plotly visualization helpers used across the CLI and Streamlit
dashboard.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------------ #
# Directory setup                                                      #
# ------------------------------------------------------------------ #
def ensure_output_dirs() -> None:
    """Create the ``outputs/`` directory tree if it does not exist.

    Creates:
        - ``outputs/plots/``
        - ``outputs/reports/``
        - ``outputs/models/``
    """
    for sub in ("plots", "reports", "models"):
        Path("outputs", sub).mkdir(parents=True, exist_ok=True)
    print("  Output directories verified.")


# ------------------------------------------------------------------ #
# Business-value accounting                                            #
# ------------------------------------------------------------------ #
def calculate_business_value(
    engine_id: int,
    dataset_name: str,
    actual_failure_cycle: Union[int, float],
    ewma_breach_cycle: Optional[int],
    ml_warning_cycle: Optional[int],
    cost_defect: float = 50_000,
    cost_false_alarm: float = 2_000,
) -> Dict:
    """Compute cost-avoidance metrics for EWMA vs ML on one engine.

    Args:
        engine_id: Engine identifier.
        dataset_name: ``"FD001"`` or ``"FD003"``.
        actual_failure_cycle: True failure cycle.
        ewma_breach_cycle: Cycle of EWMA breach (``None`` = no
            detection).
        ml_warning_cycle: Cycle of first ML warning (``None`` = no
            detection).
        cost_defect: Monetary cost of an undetected failure ($).
        cost_false_alarm: Cost of a false alarm ($).

    Returns:
        Dictionary with lead times, savings, and advantage figures.
    """
    # Lead times
    if ewma_breach_cycle is not None:
        ewma_lead = int(actual_failure_cycle - ewma_breach_cycle)
    else:
        ewma_lead = 0

    if ml_warning_cycle is not None:
        ml_lead = int(actual_failure_cycle - ml_warning_cycle)
    else:
        ml_lead = 0

    ewma_saved = cost_defect if ewma_lead > 0 else 0
    ml_saved = cost_defect if ml_lead > 0 else 0
    ml_advantage = ml_saved - ewma_saved

    return {
        "engine_id": engine_id,
        "dataset_name": dataset_name,
        "actual_failure_cycle": actual_failure_cycle,
        "ewma_breach_cycle": ewma_breach_cycle,
        "ml_warning_cycle": ml_warning_cycle,
        "ewma_lead_time": ewma_lead,
        "ml_lead_time": ml_lead,
        "ewma_value_saved": ewma_saved,
        "ml_value_saved": ml_saved,
        "ml_advantage": ml_advantage,
        "cost_defect": cost_defect,
        "cost_false_alarm": cost_false_alarm,
    }


def format_business_value_report(results: Dict) -> str:
    """Format a business-value dictionary into a printable report.

    Args:
        results: Dictionary returned by :func:`calculate_business_value`.

    Returns:
        A multi-line box-formatted string.
    """
    w = 55
    lines = [
        "+" + "-" * w + "+",
        f"|{'BUSINESS VALUE REPORT':^{w}}|",
        f"|{results['dataset_name'] + ' - Engine ' + str(results['engine_id']):^{w}}|",
        "+" + "-" * w + "+",
        f"|  Actual Failure Cycle : {str(results['actual_failure_cycle']):>{w - 27}}|",
        f"|  EWMA Breach Cycle    : {str(results['ewma_breach_cycle'] or 'N/A'):>{w - 27}}|",
        f"|  ML Warning Cycle     : {str(results['ml_warning_cycle'] or 'N/A'):>{w - 27}}|",
        "+" + "-" * w + "+",
        f"|  EWMA Lead Time       : {str(results['ewma_lead_time']) + ' cycles':>{w - 27}}|",
        f"|  ML Lead Time         : {str(results['ml_lead_time']) + ' cycles':>{w - 27}}|",
        "+" + "-" * w + "+",
        f"|  EWMA Value Saved     : ${results['ewma_value_saved']:>{w - 29},}|",
        f"|  ML Value Saved       : ${results['ml_value_saved']:>{w - 29},}|",
        f"|  ML Advantage         : ${results['ml_advantage']:>{w - 29},}|",
        "+" + "-" * w + "+",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Plotly visualisation helpers                                         #
# ------------------------------------------------------------------ #
def plot_sensor_grid_plotly(
    train_df: pd.DataFrame,
    engine_id: int,
    sensor_cols: List[str],
    dataset_name: str,
) -> go.Figure:
    """Create a subplot grid of sensor trends for a single engine.

    Args:
        train_df: Processed training DataFrame.
        engine_id: Engine to visualise.
        sensor_cols: List of sensor column names to plot.
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with one subplot per
        sensor.
    """
    df_eng = train_df[train_df["engine_id"] == engine_id]
    n = len(sensor_cols)
    cols = 4
    rows = math.ceil(n / cols)

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=sensor_cols,
        vertical_spacing=0.06,
        horizontal_spacing=0.06,
    )

    # Cycle at which RUL first drops to ≤ 15
    warning_mask = df_eng["RUL"] <= 15
    warning_cycle = int(df_eng.loc[warning_mask, "cycle"].min()) if warning_mask.any() else None

    for idx, col in enumerate(sensor_cols):
        r = idx // cols + 1
        c = idx % cols + 1
        fig.add_trace(
            go.Scatter(
                x=df_eng["cycle"], y=df_eng[col],
                mode="lines", name=col, showlegend=False,
                line=dict(width=1),
            ),
            row=r, col=c,
        )
        if warning_cycle is not None:
            fig.add_vline(
                x=warning_cycle, line_dash="dash", line_color="red",
                line_width=1, row=r, col=c,
            )

    fig.update_layout(
        title=f"All Sensor Trends - {dataset_name} Engine {engine_id}",
        height=250 * rows,
        template="plotly_white",
        showlegend=False,
    )
    return fig


def plot_rul_distribution_plotly(
    train_df: pd.DataFrame, dataset_name: str
) -> go.Figure:
    """Histogram of engine lifetimes (max cycles per engine).

    Args:
        train_df: Processed training DataFrame.
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        A Plotly histogram figure.
    """
    lifetimes = train_df.groupby("engine_id")["cycle"].max()
    fig = go.Figure(
        go.Histogram(x=lifetimes, nbinsx=30, marker_color="royalblue")
    )
    fig.update_layout(
        title=f"Engine Lifetime Distribution - {dataset_name}",
        xaxis_title="Lifetime (cycles)",
        yaxis_title="Count",
        template="plotly_white",
        height=400,
    )
    return fig


def plot_probability_timeline_plotly(
    cycle_series: pd.Series,
    proba_series: pd.Series,
    engine_id: int,
    ml_warning_cycle: Optional[int],
    actual_failure_cycle: Union[int, float],
    dataset_name: str,
) -> go.Figure:
    """Plot ML failure probability over time for a single engine.

    Args:
        cycle_series: Cycle numbers.
        proba_series: Predicted failure probabilities.
        engine_id: Engine identifier.
        ml_warning_cycle: Cycle of the first ML warning.
        actual_failure_cycle: True failure cycle.
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        A Plotly line-chart figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cycle_series, y=proba_series,
            mode="lines", name="P(failure)",
            line=dict(color="darkorange", width=2),
        )
    )

    # Decision threshold
    fig.add_hline(
        y=0.5, line_dash="dash", line_color="red",
        annotation_text="Threshold = 0.5",
    )

    # ML warning
    if ml_warning_cycle is not None:
        fig.add_vline(
            x=ml_warning_cycle, line_dash="dash", line_color="orange",
            line_width=2,
        )
        fig.add_annotation(
            x=ml_warning_cycle, y=0.5,
            text=f"ML Warning @ {ml_warning_cycle}",
            showarrow=True, arrowhead=2, font=dict(color="orange"),
        )

    # Actual failure
    fig.add_vline(
        x=actual_failure_cycle, line_dash="solid", line_color="red",
        line_width=2,
    )
    fig.add_annotation(
        x=actual_failure_cycle, y=0.9,
        text=f"Failure @ {int(actual_failure_cycle)}",
        showarrow=True, arrowhead=2, font=dict(color="red"),
    )

    fig.update_layout(
        title=f"ML Failure Probability Over Time - {dataset_name} Engine {engine_id}",
        xaxis_title="Cycle",
        yaxis_title="P(failure)",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[-0.05, 1.05]),
    )
    return fig
