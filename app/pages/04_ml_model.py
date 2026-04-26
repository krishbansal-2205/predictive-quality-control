"""
04_ml_model.py
--------------
Streamlit page: XGBoost model training, evaluation metrics, and SHAP
feature-importance chart.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import classification_report as cr
import streamlit as st

from src import data_processing, explainability, modeling

# ------------------------------------------------------------------ #
# Page config                                                          #
# ------------------------------------------------------------------ #
dataset_choice = st.session_state.get("dataset_choice", "FD001")
st.title(f"🤖 ML Model — {dataset_choice}")


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
# Model status                                                         #
# ------------------------------------------------------------------ #
model = None

if modeling.model_exists(dataset_choice):
    st.info(f"✅ Pre-trained model found for **{dataset_choice}**. Loading from disk.")
    model = cached_load_model(dataset_choice)
else:
    st.warning(
        f"⚠️ No trained model found for **{dataset_choice}**. "
        "Click **Train Model** to train now."
    )

# ------------------------------------------------------------------ #
# Train button                                                         #
# ------------------------------------------------------------------ #
if st.button("🚀 Train Model"):
    with st.spinner("Training XGBoost model…"):
        X_train, y_train = modeling.prepare_features_targets(train_df)
        model = modeling.train_model(X_train, y_train, dataset_choice)
        # Clear the cached loader so it picks up the new file
        cached_load_model.clear()
    st.success("✅ Model trained and saved!")

if model is None:
    st.stop()

# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #
st.subheader("Model Evaluation")
X_test, y_test = modeling.prepare_features_targets(test_df)
metrics = modeling.evaluate_model(
    model, X_test, y_test, dataset_choice,
    save_path=Path(f"outputs/reports/{dataset_choice}_classification_report.txt"),
)

# Reuse a single predict call — evaluate_model already called predict internally;
# we call classification_report once here for the st.code display.
y_pred = model.predict(X_test)
report_str = cr(y_test, y_pred)
st.code(report_str, language="text")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
m2.metric("Precision", f"{metrics['precision']:.3f}")
m3.metric("Recall", f"{metrics['recall']:.3f}")
m4.metric("F1 Score", f"{metrics['f1']:.3f}")

# ------------------------------------------------------------------ #
# SHAP                                                                 #
# ------------------------------------------------------------------ #
st.subheader("Feature Importance (SHAP)")

with st.spinner("Computing SHAP values…"):
    shap_values, X_sample = explainability.generate_shap_values(model, X_test, sample_size=200)

fig = explainability.plot_shap_summary_plotly(shap_values, X_sample, dataset_choice)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Hyperparameter explainer                                             #
# ------------------------------------------------------------------ #
with st.expander("📋 About XGBoost Hyperparameters"):
    st.markdown(
        """
| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 200 | Enough trees for complex patterns without overfitting |
| `max_depth` | 6 | Moderate depth balances capacity vs. generalisation |
| `learning_rate` | 0.05 | Small steps → smoother convergence |
| `subsample` | 0.8 | Row sampling adds regularisation |
| `colsample_bytree` | 0.8 | Feature sampling reduces correlation between trees |
| `eval_metric` | logloss | Standard for binary classification |
| `random_state` | 42 | Reproducibility |
"""
    )
