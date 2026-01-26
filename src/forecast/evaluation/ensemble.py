"""Ensemble methods for combining model forecasts."""

import numpy as np
import pandas as pd

from forecast.utils import get_logger


def inverse_mape_weights(
    mapes: dict[str, float],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate normalized inverse MAPE weights.

    Models with MAPE >= threshold or MAPE <= 0 are excluded.

    Args:
        mapes: Dictionary mapping model names to MAPE values.
        threshold: Maximum MAPE threshold for inclusion.

    Returns:
        Dictionary of normalized weights (sum to 1.0).
    """
    logger = get_logger()

    inverse = {}
    for name, mape_val in mapes.items():
        if 0 < mape_val < threshold:
            inverse[name] = 1.0 / mape_val
        else:
            logger.debug(
                f"Model '{name}' excluded from ensemble "
                f"(MAPE={mape_val:.3f}, threshold={threshold})"
            )

    if not inverse:
        logger.warning("No models passed MAPE threshold, using equal weights")
        valid_models = [n for n, m in mapes.items() if m > 0]
        if valid_models:
            return {n: 1.0 / len(valid_models) for n in valid_models}
        return {}

    total = sum(inverse.values())
    return {name: weight / total for name, weight in inverse.items()}


def create_ensemble(
    predictions: dict[str, pd.Series],
    metrics: dict[str, dict[str, float]],
    mape_threshold: float = 0.5,
) -> tuple[pd.Series, dict[str, float], list[str]]:
    """Create weighted ensemble forecast from multiple models.

    Args:
        predictions: Dictionary mapping model names to forecast Series.
        metrics: Dictionary mapping model names to metric dictionaries.
        mape_threshold: Maximum MAPE for model inclusion.

    Returns:
        Tuple of (ensemble_forecast, weights_used, models_used).
    """
    logger = get_logger()

    mapes = {name: m.get("mape", float("inf")) for name, m in metrics.items()}
    weights = inverse_mape_weights(mapes, mape_threshold)

    if not weights:
        logger.error("No models available for ensemble")
        return pd.Series(dtype=float), {}, []

    models_used = list(weights.keys())
    logger.info(
        f"Creating ensemble with {len(models_used)} models: "
        f"{', '.join(f'{n}({w:.2f})' for n, w in weights.items())}"
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
