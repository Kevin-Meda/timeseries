"""Time series classification module."""

import pandas as pd
import numpy as np

from forecast.utils import get_logger


def classify_series(
    series: pd.Series,
    lookback_months: int = 24,
    min_nonzero_pct: float = 0.5,
) -> bool:
    """Classify whether a time series is forecastable.

    A series is considered forecastable if it has sufficient non-zero values
    in the lookback period.

    Args:
        series: Input time series.
        lookback_months: Number of recent months to analyze.
        min_nonzero_pct: Minimum percentage of non-zero values required.

    Returns:
        True if the series is forecastable, False otherwise.
    """
    logger = get_logger()

    if len(series) == 0:
        logger.debug(f"Series '{series.name}' is empty - not forecastable")
        return False

    lookback_data = series.iloc[-lookback_months:] if len(series) >= lookback_months else series

    non_zero_count = np.sum(lookback_data != 0)
    total_count = len(lookback_data)
    non_zero_pct = non_zero_count / total_count if total_count > 0 else 0

    is_forecastable = non_zero_pct >= min_nonzero_pct

    logger.info(
        f"Classification for '{series.name}': "
        f"{non_zero_count}/{total_count} non-zero values "
        f"({non_zero_pct:.1%}) - {'Forecastable' if is_forecastable else 'Not forecastable'}"
    )

    return is_forecastable


def classify_all_series(
    series_dict: dict[str, pd.Series],
    lookback_months: int = 24,
    min_nonzero_pct: float = 0.5,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series]]:
    """Classify all series and separate forecastable from non-forecastable.

    Args:
        series_dict: Dictionary of category name to time series.
        lookback_months: Number of recent months to analyze.
        min_nonzero_pct: Minimum percentage of non-zero values required.

    Returns:
        Tuple of (forecastable_dict, non_forecastable_dict).
    """
    logger = get_logger()
    forecastable = {}
    non_forecastable = {}

    for name, series in series_dict.items():
        if classify_series(series, lookback_months, min_nonzero_pct):
            forecastable[name] = series
        else:
            non_forecastable[name] = series

    logger.info(
        f"Classification complete: {len(forecastable)} forecastable, "
        f"{len(non_forecastable)} not forecastable"
    )

    return forecastable, non_forecastable
