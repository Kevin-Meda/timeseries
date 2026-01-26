"""Time series splitting module."""

import pandas as pd

from forecast.utils import get_logger


def split_series(
    series: pd.Series,
    val_months: int = 12,
    test_months: int = 12,
    window_months: int = 60,
) -> dict:
    """Split time series into train, validation, and test sets.

    Args:
        series: Input time series with DatetimeIndex.
        val_months: Number of months for validation set.
        test_months: Number of months for test set.
        window_months: Total data window to use (most recent N months).

    Returns:
        Dictionary containing:
            - train: Training series
            - val: Validation series
            - test: Test series
            - train_idx: Indices for training data
            - val_idx: Indices for validation data
            - test_idx: Indices for test data
            - full: Full windowed series
    """
    logger = get_logger()

    if len(series) > window_months:
        series = series.iloc[-window_months:]
        logger.debug(f"Windowed series to last {window_months} months")

    total_len = len(series)
    min_required = val_months + test_months + 12

    if total_len < min_required:
        logger.warning(
            f"Series length ({total_len}) is less than minimum required "
            f"({min_required}). Adjusting splits."
        )
        test_months = min(test_months, total_len // 3)
        val_months = min(val_months, total_len // 3)

    test_start_idx = total_len - test_months
    val_start_idx = test_start_idx - val_months

    train = series.iloc[:val_start_idx]
    val = series.iloc[val_start_idx:test_start_idx]
    test = series.iloc[test_start_idx:]

    logger.debug(
        f"Split sizes - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )

    return {
        "train": train,
        "val": val,
        "test": test,
        "train_idx": series.index[:val_start_idx],
        "val_idx": series.index[val_start_idx:test_start_idx],
        "test_idx": series.index[test_start_idx:],
        "full": series,
    }
