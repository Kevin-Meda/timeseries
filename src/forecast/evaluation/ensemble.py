"""Ensemble methods for combining model forecasts."""

import numpy as np
import pandas as pd

from forecast.utils import get_logger


def inverse_mape_weights(
    mapes: dict[str, float],
    threshold: float = 0.5,
    max_models: int | None = None,
) -> dict[str, float]:
    """Calculate normalized inverse MAPE weights.

    Models with MAPE >= threshold or MAPE <= 0 are excluded.
    After threshold filtering, only the top N models (by lowest MAPE) are kept.

    Args:
        mapes: Dictionary mapping model names to MAPE values.
        threshold: Maximum MAPE threshold for inclusion.
        max_models: Maximum number of models to include. If None, include all
            that pass threshold.

    Returns:
        Dictionary of normalized weights (sum to 1.0).
    """
    logger = get_logger()

    # Filter by threshold first
    passing_models = {}
    for name, mape_val in mapes.items():
        if 0 < mape_val < threshold:
            passing_models[name] = mape_val
        else:
            logger.debug(
                f"Model '{name}' excluded from ensemble "
                f"(MAPE={mape_val:.3f}, threshold={threshold})"
            )

    if not passing_models:
        logger.warning("No models passed MAPE threshold, using equal weights")
        valid_models = [n for n, m in mapes.items() if m > 0]
        if valid_models:
            return {n: 1.0 / len(valid_models) for n in valid_models}
        return {}

    # Sort by MAPE ascending and take top N if max_models specified
    if max_models is not None and len(passing_models) > max_models:
        sorted_models = sorted(passing_models.items(), key=lambda x: x[1])
        top_models = dict(sorted_models[:max_models])
        excluded = [name for name, _ in sorted_models[max_models:]]
        logger.debug(
            f"Limiting ensemble to top {max_models} models, "
            f"excluded: {excluded}"
        )
        passing_models = top_models

    # Calculate inverse MAPE weights
    inverse = {name: 1.0 / mape_val for name, mape_val in passing_models.items()}

    total = sum(inverse.values())
    return {name: weight / total for name, weight in inverse.items()}


def create_ensemble(
    predictions: dict[str, pd.Series],
    metrics: dict[str, dict[str, float]],
    mape_threshold: float = 0.5,
    max_models: int | None = None,
) -> tuple[pd.Series, dict[str, float], list[str]]:
    """Create weighted ensemble forecast from multiple models.

    Args:
        predictions: Dictionary mapping model names to forecast Series.
        metrics: Dictionary mapping model names to metric dictionaries.
        mape_threshold: Maximum MAPE for model inclusion.
        max_models: Maximum number of models to include in ensemble.
            If None, include all that pass threshold.

    Returns:
        Tuple of (ensemble_forecast, weights_used, models_used).
    """
    logger = get_logger()

    # Only include models that have both metrics and predictions
    available_models = set(predictions.keys()) & set(metrics.keys())
    if not available_models:
        logger.error("No models have both metrics and predictions")
        return pd.Series(dtype=float), {}, []

    mapes = {name: m.get("mape", float("inf")) for name, m in metrics.items() if name in available_models}
    weights = inverse_mape_weights(mapes, mape_threshold, max_models)

    if not weights:
        logger.error("No models available for ensemble")
        return pd.Series(dtype=float), {}, []

    # Filter models_used to only those with predictions
    models_used = [m for m in weights.keys() if m in predictions]
    if not models_used:
        logger.error("No models with predictions available for ensemble")
        return pd.Series(dtype=float), {}, []

    logger.info(
        f"Creating ensemble with {len(models_used)} models: "
        f"{', '.join(f'{n}({weights[n]:.2f})' for n in models_used)}"
    )

    reference_index = predictions[models_used[0]].index
    ensemble = pd.Series(np.zeros(len(reference_index)), index=reference_index)

    for name, weight in weights.items():
        if name in predictions:
            pred = predictions[name]
            if len(pred) == len(ensemble):
                ensemble += weight * pred.values
            else:
                logger.warning(
                    f"Prediction length mismatch for {name}: "
                    f"expected {len(ensemble)}, got {len(pred)}"
                )

    ensemble.name = "Ensemble_forecast"
    return ensemble, weights, models_used


def evaluate_ensemble(
    ensemble: pd.Series,
    actual: pd.Series,
) -> dict[str, float]:
    """Evaluate ensemble forecast against actual values.

    Args:
        ensemble: Ensemble forecast series.
        actual: Actual values series.

    Returns:
        Dictionary of evaluation metrics.
    """
    from forecast.evaluation.metrics import evaluate_all

    min_len = min(len(ensemble), len(actual))
    return evaluate_all(actual.iloc[:min_len], ensemble.iloc[:min_len])
