"""Feature engineering utilities for time series forecasting."""

import numpy as np
import pandas as pd


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create lag features for the target variable.

    Args:
        df: DataFrame with target column.
        target_col: Name of the target column to create lags from.
        lags: List of lag periods. Defaults to [1, 2, 3, 6, 12].

    Returns:
        DataFrame with lag features added.
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12]

    result = df.copy()

    for lag in lags:
        result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)

    return result


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create rolling window features (mean and std).

    Uses shift(1) to avoid data leakage - only uses past values.

    Args:
        df: DataFrame with target column.
        target_col: Name of the target column.
        windows: List of window sizes. Defaults to [3, 6, 12].

    Returns:
        DataFrame with rolling features added.
    """
    if windows is None:
        windows = [3, 6, 12]

    result = df.copy()

    for window in windows:
        # Shift by 1 to avoid using current value (data leakage prevention)
        shifted = result[target_col].shift(1)
        result[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window=window).mean()
        result[f"{target_col}_roll_std_{window}"] = shifted.rolling(window=window).std()

    return result


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from the DatetimeIndex.

    Creates:
    - month: 1-12
    - quarter: 1-4
    - year: actual year
    - month_sin, month_cos: cyclical encoding of month

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with calendar features added.
    """
    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")

    # Basic calendar features
    result["month"] = result.index.month
    result["quarter"] = result.index.quarter
    result["year"] = result.index.year

    # Cyclical encoding for month (preserves continuity between Dec and Jan)
    result["month_sin"] = np.sin(2 * np.pi * result.index.month / 12)
    result["month_cos"] = np.cos(2 * np.pi * result.index.month / 12)

    return result


def create_all_temporal_features(
    df: pd.DataFrame,
    target_col: str = "demand",
    lags: list[int] | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Create all temporal features (lags, rolling, calendar).

    Args:
        df: DataFrame with target column and DatetimeIndex.
        target_col: Name of the target column.
        lags: List of lag periods.
        windows: List of rolling window sizes.

    Returns:
        DataFrame with all temporal features added.
    """
    result = create_lag_features(df, target_col, lags)
    result = create_rolling_features(result, target_col, windows)
    result = create_calendar_features(result)
    return result


def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN values (typically from lagging).

    Args:
        df: DataFrame potentially containing NaN values.

    Returns:
        DataFrame with NaN rows removed.
    """
    return df.dropna()


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str = "demand",
    exclude_cols: list[str] | None = None,
) -> list[str]:
    """Get list of feature column names (excluding target and specified columns).

    Args:
        df: DataFrame with features.
        target_col: Name of the target column to exclude.
        exclude_cols: Additional columns to exclude.

    Returns:
        List of feature column names.
    """
    exclude = {target_col}
    if exclude_cols:
        exclude.update(exclude_cols)

    return [col for col in df.columns if col not in exclude]
