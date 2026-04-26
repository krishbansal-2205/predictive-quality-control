"""
data_processing.py
------------------
Unified data loading and feature engineering pipeline for NASA C-MAPSS
FD001 and FD003 datasets.  Both datasets share an identical column layout
and operate under **1 operating condition**, so no conditional
normalization is required.
"""

from pathlib import Path
from typing import Tuple, List

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
        FileNotFoundError: If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath, sep=r"\s+", header=None, engine="python")
    df.columns = _COLUMN_NAMES[: df.shape[1]]
    return df


# ------------------------------------------------------------------ #
# Cleaning                                                             #
# ------------------------------------------------------------------ #
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop operational-setting columns and zero-variance sensors.

    Args:
        df: Raw DataFrame returned by :func:`load_data`.

    Returns:
        Cleaned DataFrame with uninformative columns removed.
    """
    df = df.drop(columns=["setting_1", "setting_2", "setting_3"], errors="ignore")

    # Identify zero-variance columns dynamically
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var_cols = [c for c in numeric_cols if df[c].std() == 0]

    if zero_var_cols:
        print(f"  Dropping zero-variance columns: {zero_var_cols}")
        df = df.drop(columns=zero_var_cols)

    return df


# ------------------------------------------------------------------ #
# RUL labelling                                                        #
# ------------------------------------------------------------------ #
def add_rul_train(df: pd.DataFrame, clip_value: int = 125) -> pd.DataFrame:
    """Add clipped Remaining Useful Life (RUL) to **training** data.

    Args:
        df: Cleaned training DataFrame.
        clip_value: Maximum RUL value (piecewise-linear assumption).

    Returns:
        DataFrame with an added ``RUL`` column.
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    df["RUL"] = df["RUL"].clip(upper=clip_value)
    return df


def add_rul_test(
    df: pd.DataFrame, rul_filepath: Path, clip_value: int = 125
) -> pd.DataFrame:
    """Add clipped RUL to **test** data using the ground-truth RUL file.

    Args:
        df: Cleaned test DataFrame.
        rul_filepath: Path to the ``RUL_FDxxx.txt`` file.
        clip_value: Maximum RUL value.

    Returns:
        DataFrame with an added ``RUL`` column.

    Raises:
        FileNotFoundError: If *rul_filepath* does not exist.
    """
    rul_filepath = Path(rul_filepath)
    if not rul_filepath.exists():
        raise FileNotFoundError(f"RUL file not found: {rul_filepath}")

    df = df.copy()
    rul_df = pd.read_csv(rul_filepath, sep=r"\s+", header=None, engine="python")
    rul_df.columns = ["true_RUL"]
    rul_df["engine_id"] = range(1, len(rul_df) + 1)

    df = df.merge(rul_df, on="engine_id", how="left")
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = df["true_RUL"] + (max_cycle - df["cycle"])
    df["RUL"] = df["RUL"].clip(upper=clip_value)
    df = df.drop(columns=["true_RUL"])
    return df


# ------------------------------------------------------------------ #
# Target creation                                                      #
# ------------------------------------------------------------------ #
def create_target(df: pd.DataFrame, warning_window: int = 15) -> pd.DataFrame:
    """Create a binary classification target for early-warning detection.

    Args:
        df: DataFrame with a ``RUL`` column.
        warning_window: Number of cycles before failure that count as
            *imminent failure*.

    Returns:
        DataFrame with an added ``failure_within_window`` column.
    """
    df = df.copy()
    df["failure_within_window"] = (df["RUL"] <= warning_window).astype(int)
    return df


# ------------------------------------------------------------------ #
# Feature engineering                                                  #
# ------------------------------------------------------------------ #
def engineer_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Add rolling mean and rolling std features for every sensor column.

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

    # Back-fill any remaining NaNs within each engine group
    df = df.groupby("engine_id", group_keys=False).apply(
        lambda g: g.bfill().ffill(), include_groups=False
    )
    # Re-insert engine_id since include_groups=False drops it
    # Actually, include_groups=False only excludes the grouping column from
    # the lambda input but the result index is preserved.  However the
    # column is now missing — re-add it from the index.
    if "engine_id" not in df.columns:
        df = df.reset_index(level="engine_id")
    return df


# ------------------------------------------------------------------ #
# Full pipeline                                                        #
# ------------------------------------------------------------------ #
def prepare_dataset(
    dataset_name: str, clip_value: int = 125, warning_window: int = 15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete load → clean → label → feature pipeline.

    Args:
        dataset_name: ``"FD001"`` or ``"FD003"``.
        clip_value: RUL clipping value.
        warning_window: Warning-window width in cycles.

    Returns:
        A ``(train_df, test_df)`` tuple of fully-processed DataFrames.

    Raises:
        ValueError: If *dataset_name* is not one of the supported
            datasets.
    """
    if dataset_name not in ("FD001", "FD003"):
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choose 'FD001' or 'FD003'."
        )

    base = Path("dataset")
    train_path = base / f"train_{dataset_name}.txt"
    test_path = base / f"test_{dataset_name}.txt"
    rul_path = base / f"RUL_{dataset_name}.txt"

    print(f"\n{'-' * 50}")
    print(f"  Preparing {dataset_name}")
    print(f"{'-' * 50}")

    # Train
    train_df = load_data(train_path)
    train_df = clean_data(train_df)
    train_df = add_rul_train(train_df, clip_value)
    train_df = create_target(train_df, warning_window)
    train_df = engineer_features(train_df)

    # Test
    test_df = load_data(test_path)
    test_df = clean_data(test_df)
    test_df = add_rul_test(test_df, rul_path, clip_value)
    test_df = create_target(test_df, warning_window)
    test_df = engineer_features(test_df)

    print(f"  Train shape: {train_df.shape}")
    print(f"  Test  shape: {test_df.shape}")
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
        c
        for c in df.columns
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
