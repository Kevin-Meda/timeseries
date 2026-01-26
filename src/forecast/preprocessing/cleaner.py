"""Preprocessing and cleaning module for time series data."""

import pandas as pd
import numpy as np

from forecast.utils import get_logger


def neighbor_mean(series: pd.Series, idx: int) -> float:
    """Calculate mean of adjacent values.

    Args:
        series: Input time series.
        idx: Index of the value to calculate neighbors for.

    Returns:
        Mean of adjacent values, or 0.0 if both neighbors unavailable/invalid.
    """
    prev_val = series.iloc[idx - 1] if idx > 0 else None
    next_val = series.iloc[idx + 1] if idx < len(series) - 1 else None

    valid = []
    for v in [prev_val, next_val]:
        if v is not None and not np.isnan(v) and v >= 0:
            valid.append(v)

    return np.mean(valid) if valid else 0.0


def preprocess_series(
    series: pd.Series,
    config: dict,
) -> tuple[pd.Series, dict]:
    """Preprocess a time series by cleaning invalid values.

    Preprocessing steps:
    1. Replace negative values with neighbor mean
    2. Replace NaN values with neighbor mean
    3. Detect and replace outliers (> std_threshold std from mean)

    Args:
        series: Input time series.
        config: Configuration dict with keys:
            - outlier_std_threshold: Number of standard deviations for outlier detection

    Returns:
        Tuple of (cleaned_series, change_log).
        change_log contains lists of indices for each type of replacement.
    """
    logger = get_logger()
    cleaned = series.copy()

    change_log = {
        "negative_indices": [],
        "nan_indices": [],
        "outlier_indices": [],
        "original_values": {},
        "new_values": {},
    }

    std_threshold = config.get("outlier_std_threshold", 3.0)

    for idx in range(len(cleaned)):
        if cleaned.iloc[idx] < 0:
            original = cleaned.iloc[idx]
            new_val = neighbor_mean(cleaned, idx)
            change_log["negative_indices"].append(idx)
            change_log["original_values"][idx] = original
            change_log["new_values"][idx] = new_val
            cleaned.iloc[idx] = new_val

    for idx in range(len(cleaned)):
        if pd.isna(cleaned.iloc[idx]):
            change_log["nan_indices"].append(idx)
            change_log["original_values"][idx] = np.nan
            new_val = neighbor_mean(cleaned, idx)
            change_log["new_values"][idx] = new_val
            cleaned.iloc[idx] = new_val

    # Detect outliers using IQR method (more robust for trending data)
    valid_values = cleaned[~pd.isna(cleaned)]
    if len(valid_values) > 0:
        # Use first differences to detect outliers relative to trend
        diffs = cleaned.diff().dropna()
        if len(diffs) > 3:
            q1 = diffs.quantile(0.25)
            q3 = diffs.quantile(0.75)
            iqr = q3 - q1
            iqr_multiplier = std_threshold  # Reuse threshold parameter

            lower_diff = q1 - iqr_multiplier * iqr
            upper_diff = q3 + iqr_multiplier * iqr

            for idx in range(1, len(cleaned)):
                diff_val = cleaned.iloc[idx] - cleaned.iloc[idx - 1]
                if diff_val < lower_diff or diff_val > upper_diff:
                    if idx not in change_log["original_values"]:
                        original = cleaned.iloc[idx]
                        new_val = neighbor_mean(cleaned, idx)
                        change_log["outlier_indices"].append(idx)
                        change_log["original_values"][idx] = original
                        change_log["new_values"][idx] = new_val
                        cleaned.iloc[idx] = new_val

    total_changes = (
        len(change_log["negative_indices"])
        + len(change_log["nan_indices"])
        + len(change_log["outlier_indices"])
    )

    logger.info(
        f"Preprocessing '{series.name}': {total_changes} values replaced "
        f"(negatives: {len(change_log['negative_indices'])}, "
        f"NaN: {len(change_log['nan_indices'])}, "
        f"outliers: {len(change_log['outlier_indices'])})"
    )

    return cleaned, change_log


def preprocess_all_series(
    series_dict: dict[str, pd.Series],
    config: dict,
) -> tuple[dict[str, pd.Series], dict[str, dict]]:
    """Preprocess all series in a dictionary.

    Args:
        series_dict: Dictionary of category name to time series.
        config: Preprocessing configuration.

    Returns:
        Tuple of (cleaned_series_dict, change_logs_dict).
    """
    cleaned_dict = {}
    change_logs = {}

    for name, series in series_dict.items():
        cleaned, log = preprocess_series(series, config)
        cleaned_dict[name] = cleaned
        change_logs[name] = log

    return cleaned_dict, change_logs
