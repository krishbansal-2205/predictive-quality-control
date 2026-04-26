"""
data_processing.py
------------------
Unified data loading and feature engineering pipeline for NASA C-MAPSS
FD001 and FD003 datasets. Both datasets share an identical column layout
and operate under 1 operating condition, so no conditional
normalization is required.

Critical design decision: zero-variance columns are computed ONLY from
the training set and the same list is applied to the test set. This
guarantees train and test always have identical feature columns.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# Column names (same for every C-MAPSS sub-dataset)                    #
# ------------------------------------------------------------------ #
_COLUMN_NAMES: List[str] = (
    ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ------------------------------------------------------------------ #
# Loading                                                              #
# ------------------------------------------------------------------ #
def load_data(filepath: Path) -> pd.DataFrame:
    """Read a raw C-MAPSS text file into a DataFrame.

    Args:
        filepath: Path to the whitespace-delimited text file.

    Returns:
        DataFrame with named columns for engine_id, cycle, settings,
        and 21 sensor readings.

    Raises:
        FileNotFoundError: If filepath does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(
        filepath, sep=r"\s+", header=None, engine="python"
    )
    df.columns = _COLUMN_NAMES[: df.shape[1]]
    return df


# ------------------------------------------------------------------ #
# Cleaning                                                             #
# ------------------------------------------------------------------ #
def clean_data(
    df: pd.DataFrame,
    columns_to_drop: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop operational-setting columns and zero-variance sensors.

    The zero-variance columns are computed from the DataFrame passed in
    (intended to be the training set). The same list must then be
    passed as ``columns_to_drop`` when cleaning the test set so that
    both sets end up with identical columns.

    Args:
        df: Raw DataFrame returned by :func:`load_data`.
        columns_to_drop: If ``None``, compute zero-variance columns
            from ``df`` (use for training data). If a list, drop
            exactly those columns (use for test data, passing the list
            returned from the training call).

    Returns:
        A ``(cleaned_df, dropped_cols)`` tuple where ``dropped_cols``
        is the list of columns that were removed.
    """
    df = df.drop(
        columns=["setting_1", "setting_2", "setting_3"], errors="ignore"
    )

    if columns_to_drop is None:
        # Determine zero-variance columns from this DataFrame (train)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        columns_to_drop = [
            c for c in numeric_cols
            if c not in ("engine_id", "cycle") and df[c].std(ddof=0) == 0
        ]
        if columns_to_drop:
            print(f"  Dropping columns: {columns_to_drop}")

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")

    return df, columns_to_drop


# ------------------------------------------------------------------ #
# RUL labelling                                                        #
# ------------------------------------------------------------------ #
def add_rul_train(
    df: pd.DataFrame, clip_value: int = 125
) -> pd.DataFrame:
    """Add clipped Remaining Useful Life (RUL) to training data.

    Args:
        df: Cleaned training DataFrame.
        clip_value: Maximum RUL value (piecewise-linear assumption).

    Returns:
        DataFrame with an added ``RUL`` column.
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=clip_value)
    return df


def add_rul_test(
    df: pd.DataFrame,
    rul_filepath: Path,
    clip_value: int = 125,
) -> pd.DataFrame:
    """Add clipped RUL to test data using the ground-truth RUL file.

    Args:
        df: Cleaned test DataFrame.
        rul_filepath: Path to the ``RUL_FDxxx.txt`` file.
        clip_value: Maximum RUL value.

    Returns:
        DataFrame with an added ``RUL`` column.

    Raises:
        FileNotFoundError: If rul_filepath does not exist.
    """
    rul_filepath = Path(rul_filepath)
    if not rul_filepath.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_filepath}")

    df = df.copy()
    rul_df = pd.read_csv(
        rul_filepath, sep=r"\s+", header=None, engine="python"
    )
    rul_df.columns = ["true_RUL"]
    rul_df["engine_id"] = range(1, len(rul_df) + 1)

    df = df.merge(rul_df, on="engine_id", how="left")
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = (df["true_RUL"] + (max_cycle - df["cycle"])).clip(
        upper=clip_value
    )
    # Drop the merge helper column so it does not leak into the feature matrix.
    df = df.drop(columns=["true_RUL"])
    return df


# ------------------------------------------------------------------ #
# Target creation                                                      #
# ------------------------------------------------------------------ #
def create_target(
    df: pd.DataFrame, warning_window: int = 30
) -> pd.DataFrame:
    """Create a binary classification target for early-warning detection.

    Args:
        df: DataFrame with a ``RUL`` column.
        warning_window: Cycles before failure that count as imminent.

    Returns:
        DataFrame with an added ``failure_within_window`` column.
    """
    df = df.copy()
    df["failure_within_window"] = (df["RUL"] <= warning_window).astype(int)
    return df


# ------------------------------------------------------------------ #
# Feature engineering                                                  #
# ------------------------------------------------------------------ #
def engineer_features(
    df: pd.DataFrame, window: int = 5
) -> pd.DataFrame:
    """Add rolling mean and rolling std features for every sensor column.

    Uses ``groupby + transform`` exclusively to avoid any index
    manipulation that could drop or misalign the ``engine_id`` column.

    Args:
        df: DataFrame with sensor columns.
        window: Rolling window size.

    Returns:
        DataFrame augmented with ``{sensor}_roll_mean`` and
        ``{sensor}_roll_std`` columns.
    """
    df = df.copy()
    sensor_cols = get_sensor_columns(df)

    for col in sensor_cols:
        grouped = df.groupby("engine_id")[col]

        df[f"{col}_roll_mean"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = grouped.transform(
            lambda s: s.rolling(window=window, min_periods=1).std()
        )

    # Fill NaNs in roll_std columns (first row of each engine has NaN std
    # because std is undefined for a window of size 1).
    # A single vectorised fillna suffices — no need to re-group by engine.
    roll_std_cols = [f"{col}_roll_std" for col in sensor_cols]
    df[roll_std_cols] = df[roll_std_cols].fillna(0)

    return df


# ------------------------------------------------------------------ #
# Full pipeline                                                        #
# ------------------------------------------------------------------ #
def prepare_dataset(
    dataset_name: str,
    clip_value: int = 125,
    warning_window: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete load → clean → label → feature pipeline.

    Zero-variance columns are identified from the TRAINING set only
    and the same columns are dropped from the test set, guaranteeing
    that both DataFrames have identical feature columns.

    Args:
        dataset_name: ``"FD001"`` or ``"FD003"``.
        clip_value: RUL clipping value.
        warning_window: Warning-window width in cycles.

    Returns:
        A ``(train_df, test_df)`` tuple of fully-processed DataFrames
        with guaranteed identical feature columns.

    Raises:
        ValueError: If dataset_name is not ``"FD001"`` or ``"FD003"``.
        RuntimeError: If train and test end up with different columns
            despite the fix (safety net).
    """
    if dataset_name not in ("FD001", "FD003"):
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. "
            "Choose 'FD001' or 'FD003'."
        )

    base = Path("dataset")
    train_path = base / f"train_{dataset_name}.txt"
    test_path = base / f"test_{dataset_name}.txt"
    rul_path = base / f"RUL_{dataset_name}.txt"

    print(f"\n{'-' * 50}")
    print(f"  Preparing {dataset_name}")
    print(f"{'-' * 50}")

    # ---- Train -------------------------------------------------------
    train_df = load_data(train_path)
    # columns_to_drop=None → compute from train
    train_df, cols_to_drop = clean_data(train_df, columns_to_drop=None)
    train_df = add_rul_train(train_df, clip_value)
    train_df = create_target(train_df, warning_window)
    train_df = engineer_features(train_df)

    # ---- Test --------------------------------------------------------
    test_df = load_data(test_path)
    # Pass the SAME cols_to_drop → test matches train exactly
    test_df, _ = clean_data(test_df, columns_to_drop=cols_to_drop)
    test_df = add_rul_test(test_df, rul_path, clip_value)
    test_df = create_target(test_df, warning_window)
    test_df = engineer_features(test_df)

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test  shape: {test_df.shape}")

    # ---- Safety assertion --------------------------------------------
    train_cols = set(train_df.columns)
    test_cols = set(test_df.columns)
    only_train = train_cols - test_cols
    only_test = test_cols - train_cols
    if only_train or only_test:
        raise RuntimeError(
            f"Column mismatch after processing {dataset_name}!\n"
            f"  Only in train: {sorted(only_train)}\n"
            f"  Only in test:  {sorted(only_test)}"
        )
    print(f"  Column alignment: OK ({train_df.shape[1]} columns)")
    return train_df, test_df


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
def get_sensor_columns(df: pd.DataFrame) -> List[str]:
    """Return raw sensor column names (excluding rolling features).

    Args:
        df: Any processed DataFrame.

    Returns:
        List of column names that start with ``sensor_`` and do not
        contain ``_roll_``.
    """
    return [
        c for c in df.columns
        if c.startswith("sensor_") and "_roll_" not in c
    ]


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all feature columns suitable for modelling.

    Args:
        df: Processed DataFrame.

    Returns:
        List of column names excluding identifiers and target columns.
    """
    exclude = {"engine_id", "cycle", "RUL", "failure_within_window"}
    return [c for c in df.columns if c not in exclude]
