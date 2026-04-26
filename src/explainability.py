"""
explainability.py
-----------------
SHAP-based model explainability utilities.  Provides both Matplotlib
(CLI) and Plotly (Streamlit) visualisations for feature importance.
"""

from pathlib import Path
from typing import Any, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap


# ------------------------------------------------------------------ #
# SHAP value generation                                                #
# ------------------------------------------------------------------ #
def generate_shap_values(
    model: Any, X_test: pd.DataFrame, sample_size: int = 200
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Compute SHAP values for a sample of the test set.

    Args:
        model: A fitted tree-based classifier (e.g. XGBoost).
        X_test: Full test feature matrix.
        sample_size: Number of rows to explain.

    Returns:
        A ``(shap_values, X_sample)`` tuple where *shap_values* is a
        NumPy array of shape ``(sample_size, n_features)`` and
        *X_sample* is the corresponding feature DataFrame.
    """
    sample_size = min(sample_size, len(X_test))
    X_sample = X_test.iloc[:sample_size].copy()
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    # Normalise: some SHAP versions return a list [class0, class1] for
    # binary classifiers. Always extract the class-1 (positive) SHAP array.
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return shap_vals, X_sample


# ------------------------------------------------------------------ #
# Matplotlib plot (CLI / notebook)                                     #
# ------------------------------------------------------------------ #
def plot_shap_summary_matplotlib(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    dataset_name: str,
    save_path: Path,
) -> None:
    """Save a SHAP bee-swarm summary plot via Matplotlib.

    Args:
        shap_values: SHAP values array.
        X_sample: Feature DataFrame matching *shap_values*.
        dataset_name: ``"FD001"`` or ``"FD003"``.
        save_path: Destination PNG path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title(f"SHAP Feature Importance - {dataset_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP summary plot saved -> {save_path}")


# ------------------------------------------------------------------ #
# Plotly plot (Streamlit dashboard)                                    #
# ------------------------------------------------------------------ #
def plot_shap_summary_plotly(
    shap_values: np.ndarray,
    X_sample: pd.DataFrame,
    dataset_name: str,
) -> go.Figure:
    """Build an interactive Plotly bar chart of SHAP feature importance.

    Args:
        shap_values: SHAP values array.
        X_sample: Feature DataFrame matching *shap_values*.
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        A :class:`plotly.graph_objects.Figure` with horizontal bars for
        the top 15 features.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {"feature": X_sample.columns, "mean_abs_shap": mean_abs}
    ).sort_values("mean_abs_shap", ascending=True)

    top = importance_df.tail(15)

    fig = go.Figure(
        go.Bar(
            x=top["mean_abs_shap"],
            y=top["feature"],
            orientation="h",
            marker_color="royalblue",
        )
    )
    fig.update_layout(
        title=f"Top 15 Features by SHAP Importance - {dataset_name}",
        xaxis_title="Mean |SHAP value|",
        yaxis_title="Feature",
        template="plotly_white",
        height=500,
        margin=dict(l=180),
    )
    return fig


# ------------------------------------------------------------------ #
# SHAP DataFrame helper                                                #
# ------------------------------------------------------------------ #
def get_shap_dataframe(
    shap_values: np.ndarray, X_sample: pd.DataFrame
) -> pd.DataFrame:
    """Convert SHAP values into a tidy DataFrame.

    Args:
        shap_values: SHAP values array.
        X_sample: Feature DataFrame matching *shap_values*.

    Returns:
        DataFrame with one column per feature containing the raw SHAP
        value for that sample, plus a ``mean_abs_shap`` column holding
        the mean absolute SHAP value across all features for each sample
        (a per-row overall importance score).
    """
    df = pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)
    df["mean_abs_shap"] = np.abs(shap_values).mean(axis=1)
    return df
