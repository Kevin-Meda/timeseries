"""Time series splitting module."""

import pandas as pd

from forecast.utils import get_logger


def split_series(
    data: pd.Series | pd.DataFrame,
    val_months: int = 12,
    test_months: int = 12,
    window_months: int = 60,
) -> dict:
    """Split time series into train, validation, and test sets.

    Args:
        data: Input time series (Series) or DataFrame with DatetimeIndex.
            If DataFrame, expected to have a 'demand' column as target.
        val_months: Number of months for validation set.
        test_months: Number of months for test set.
        window_months: Total data window to use (most recent N months).

    Returns:
        Dictionary containing:
            - train: Training data (Series or DataFrame)
            - val: Validation data (Series or DataFrame)
            - test: Test data (Series or DataFrame)
            - train_idx: Indices for training data
            - val_idx: Indices for validation data
            - test_idx: Indices for test data
            - full: Full windowed data
    """
    logger = get_logger()

    # Handle both Series and DataFrame
    is_dataframe = isinstance(data, pd.DataFrame)

    if len(data) > window_months:
        data = data.iloc[-window_months:]
        logger.debug(f"Windowed data to last {window_months} months")

    total_len = len(data)
    min_required = val_months + test_months + 12

    if total_len < min_required:
        logger.warning(
            f"Data length ({total_len}) is less than minimum required "
            f"({min_required}). Adjusting splits."
        )
        test_months = min(test_months, total_len // 3)
        val_months = min(val_months, total_len // 3)

    test_start_idx = total_len - test_months
    val_start_idx = test_start_idx - val_months

    train = data.iloc[:val_start_idx]
    val = data.iloc[val_start_idx:test_start_idx]
    test = data.iloc[test_start_idx:]

    logger.debug(
        f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_idx": data.index[:val_start_idx],
        "val_idx": data.index[val_start_idx:test_start_idx],
        "test_idx": data.index[test_start_idx:],
        "full": data,
    }


def split_with_features(
    demand_series: pd.Series,
    features_df: pd.DataFrame | None,
    val_months: int = 12,
    test_months: int = 12,
    window_months: int = 60,
) -> dict:
    """Split time series and aligned features into train/val/test.

    Args:
        demand_series: Demand time series with DatetimeIndex.
        features_df: Optional DataFrame with features (same index as demand).
        val_months: Number of months for validation set.
        test_months: Number of months for test set.
        window_months: Total data window to use.

    Returns:
        Dictionary containing splits for both demand and features.
    """
    # Create combined DataFrame if features provided
    if features_df is not None and len(features_df) > 0:
        combined = pd.DataFrame({"demand": demand_series})
        for col in features_df.columns:
            if col in combined.index or col == "demand":
                continue
            combined[col] = features_df[col]
        data = combined
    else:
        data = demand_series

    return split_series(data, val_months, test_months, window_months)
