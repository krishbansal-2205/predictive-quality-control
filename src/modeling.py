"""
modeling.py
-----------
XGBoost classification model training, evaluation, persistence, and
prediction utilities.  Models for FD001 and FD003 are trained and saved
separately using ``joblib``.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier


# ------------------------------------------------------------------ #
# Feature / target split                                               #
# ------------------------------------------------------------------ #
def prepare_features_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame into features and binary target.

    Args:
        df: Processed DataFrame containing at least
            ``failure_within_window``.

    Returns:
        A ``(X, y)`` tuple where *X* is the feature matrix and *y* is
        the binary target series.

    Raises:
        KeyError: If ``failure_within_window`` is missing.
    """
    drop_cols = ["engine_id", "cycle", "RUL", "failure_within_window"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["failure_within_window"]
    return X, y


# ------------------------------------------------------------------ #
# Training                                                             #
# ------------------------------------------------------------------ #
def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, dataset_name: str
) -> XGBClassifier:
    """Train an XGBoost classifier and persist it to disk.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        dataset_name: ``"FD001"`` or ``"FD003"`` — used for the
            filename.

    Returns:
        The fitted :class:`XGBClassifier`.
    """
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    save_dir = Path("outputs/models")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{dataset_name}_xgb_model.joblib"
    joblib.dump(model, save_path)
    print(f"  Model trained and saved for {dataset_name} -> {save_path}")
    return model


# ------------------------------------------------------------------ #
# Persistence helpers                                                  #
# ------------------------------------------------------------------ #
def load_model(dataset_name: str) -> XGBClassifier:
    """Load a previously saved XGBoost model from disk.

    Args:
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        The loaded :class:`XGBClassifier`.

    Raises:
        FileNotFoundError: If no saved model exists for the given
            dataset.
    """
    path = Path(f"outputs/models/{dataset_name}_xgb_model.joblib")
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at {path}. "
            f"Train one first with `train_model()` or via main.py."
        )
    return joblib.load(path)


def model_exists(dataset_name: str) -> bool:
    """Check whether a saved model exists on disk.

    Args:
        dataset_name: ``"FD001"`` or ``"FD003"``.

    Returns:
        ``True`` if the model file exists, ``False`` otherwise.
    """
    return Path(f"outputs/models/{dataset_name}_xgb_model.joblib").exists()


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #
def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dataset_name: str,
    save_path: Path,
) -> Dict[str, float]:
    """Evaluate a trained model and save the classification report.

    Args:
        model: Fitted classifier.
        X_test: Test feature matrix.
        y_test: Test target vector.
        dataset_name: Dataset identifier.
        save_path: File path for the text report.

    Returns:
        Dictionary with ``accuracy``, ``precision``, ``recall``, and
        ``f1`` (all for class 1).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(f"\n  Classification Report - {dataset_name}")
    print(report)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {dataset_name}\n")
        f.write("=" * 55 + "\n")
        f.write(report)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }


# ------------------------------------------------------------------ #
# Prediction helpers                                                   #
# ------------------------------------------------------------------ #
def predict_proba_series(
    model: XGBClassifier, df_engine: pd.DataFrame
) -> pd.Series:
    """Return failure probabilities for every row of a single engine.

    Args:
        model: Fitted classifier.
        df_engine: Processed DataFrame for a single engine.

    Returns:
        A pd.Series of P(failure) with the ORIGINAL index of
        df_engine preserved so that .loc[] lookups work correctly.
    """
    drop_cols = ["engine_id", "cycle", "RUL", "failure_within_window"]
    drop_cols = [c for c in drop_cols if c in df_engine.columns]
    X = df_engine.drop(columns=drop_cols)
    proba = model.predict_proba(X)[:, 1]
    # CRITICAL: use df_engine.index, not a reset index
    return pd.Series(proba, index=df_engine.index, name="failure_proba")


def predict_failure_start(
    model: XGBClassifier,
    df_engine: pd.DataFrame,
    threshold: float = 0.3,
) -> Tuple[Optional[int], pd.Series]:
    """Predict the first cycle at which failure probability exceeds threshold.

    Args:
        model: Fitted classifier.
        df_engine: Processed DataFrame for a single engine.
        threshold: Decision threshold on the predicted probability.

    Returns:
        A ``(first_warning_cycle, proba_series)`` tuple. The warning
        cycle is ``None`` if the threshold is never exceeded.
    """
    # Reset index on a copy so positional and label indexing are aligned
    df_reset = df_engine.reset_index(drop=True)
    proba = predict_proba_series(model, df_reset)

    above = proba[proba > threshold]
    if above.empty:
        return None, proba

    # above.index[0] is now a clean integer positional index
    first_warning_cycle = int(df_reset.loc[above.index[0], "cycle"])
    return first_warning_cycle, proba
