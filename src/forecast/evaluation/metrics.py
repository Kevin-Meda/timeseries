"""Evaluation metrics for forecasting models."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error as sklearn_mae


def rmse(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Calculate Root Mean Square Error.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        RMSE value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Arrays must have the same length")

    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Calculate Mean Absolute Percentage Error.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        MAPE value (as decimal, not percentage).
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Arrays must have the same length")

    mask = actual != 0
    if not np.any(mask):
        return 0.0

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def mae_score(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Calculate Mean Absolute Error.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        MAE score.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if len(actual) != len(predicted):
        raise ValueError("Arrays must have the same length")

    if len(actual) < 1:
        return 0.0

    return float(sklearn_mae(actual, predicted))


def evaluate_all(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Calculate all evaluation metrics.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        Dictionary with RMSE, MAPE, and MAE scores.
    """
    return {
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "mae": mae_score(actual, predicted),
    }


def calculate_aggregate_metrics(metrics_dict: dict[str, dict[str, float]]) -> dict[str, float]:
    """Calculate aggregate statistics across multiple categories.

    Args:
        metrics_dict: Dictionary mapping category names to metric dictionaries.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if not metrics_dict:
        return {}

    metric_names = list(next(iter(metrics_dict.values())).keys())
    result = {}

    for metric in metric_names:
        values = [m[metric] for m in metrics_dict.values() if metric in m]
        if values:
            result[f"{metric}_mean"] = float(np.mean(values))
            result[f"{metric}_std"] = float(np.std(values))

    return result
