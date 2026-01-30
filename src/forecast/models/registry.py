"""Model registry for creating forecasters based on configuration."""

from typing import Any

from forecast.models.base import BaseForecaster
from forecast.models.sarima import SARIMAForecaster
from forecast.models.holt_winters import HoltWintersForecaster
from forecast.models.chronos import ChronosForecaster, is_chronos_available
from forecast.models.xgboost import XGBoostForecaster, is_xgboost_available
from forecast.models.prophet import ProphetForecaster, is_prophet_available
from forecast.utils import get_logger


def create_models(
    config: dict[str, Any],
    product_mode: str = "per_product",
) -> list[BaseForecaster]:
    """Create forecaster instances based on configuration.

    Args:
        config: Models configuration dictionary.
        product_mode: "per_product" or "multi_product".

    Returns:
        List of enabled and available forecaster instances.
    """
    logger = get_logger()
    models: list[BaseForecaster] = []
    retrain = config.get("retrain", True)

    # SARIMA
    sarima_config = config.get("sarima", {})
    if sarima_config.get("enabled", False):
        use_optuna = retrain and sarima_config.get("use_optuna", False)
        optuna_trials = sarima_config.get("optuna_trials", 50)
        models.append(
            SARIMAForecaster(
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
            )
        )
        logger.debug(f"Created SARIMA forecaster (optuna={use_optuna})")

    # Holt-Winters
    hw_config = config.get("holt_winters", {})
    if hw_config.get("enabled", False):
        trend = hw_config.get("trend", "add")
        seasonal = hw_config.get("seasonal", "add")
        seasonal_periods = hw_config.get("seasonal_periods", 12)
        models.append(
            HoltWintersForecaster(
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
            )
        )
        logger.debug("Created Holt-Winters forecaster")

    # Chronos
    chronos_config = config.get("chronos", {})
    if chronos_config.get("enabled", False):
        if is_chronos_available():
            model_name = chronos_config.get("model_name", "amazon/chronos-2")
            models.append(ChronosForecaster(model_name=model_name))
            logger.debug("Created Chronos forecaster")
        else:
            logger.warning("Chronos enabled but dependencies not available")

    # XGBoost
    xgb_config = config.get("xgboost", {})
    if xgb_config.get("enabled", False):
        if is_xgboost_available():
            use_optuna = retrain
            defaults = xgb_config.get("defaults", {})
            models.append(
                XGBoostForecaster(
                    use_optuna=use_optuna,
                    optuna_trials=xgb_config.get("optuna_trials", 100),
                    n_estimators=defaults.get("n_estimators", 100),
                    max_depth=defaults.get("max_depth", 6),
                    learning_rate=defaults.get("learning_rate", 0.1),
                )
            )
            logger.debug(f"Created XGBoost forecaster (optuna={use_optuna})")
        else:
            logger.warning("XGBoost enabled but not available")

    # Prophet
    prophet_config = config.get("prophet", {})
    if prophet_config.get("enabled", False):
        if is_prophet_available():
            use_optuna = retrain
            defaults = prophet_config.get("defaults", {})
            models.append(
                ProphetForecaster(
                    use_optuna=use_optuna,
                    optuna_trials=prophet_config.get("optuna_trials", 50),
                    changepoint_prior_scale=defaults.get("changepoint_prior_scale", 0.05),
                    seasonality_prior_scale=defaults.get("seasonality_prior_scale", 10.0),
                )
            )
            logger.debug(f"Created Prophet forecaster (optuna={use_optuna})")
        else:
            logger.warning("Prophet enabled but not available")

    logger.info(f"Created {len(models)} forecasters: {[m.name for m in models]}")
    return models


def get_available_models() -> dict[str, bool]:
    """Get availability status of all models.

    Returns:
        Dictionary mapping model names to availability status.
    """
    return {
        "SARIMA": True,
        "HoltWinters": True,
        "Chronos": is_chronos_available(),
        "XGBoost": is_xgboost_available(),
        "Prophet": is_prophet_available(),
    }


def get_multivariate_models(models: list[BaseForecaster]) -> list[BaseForecaster]:
    """Filter models that support multivariate input.

    Args:
        models: List of forecaster instances.

    Returns:
        List of forecasters that support exogenous features.
    """
    return [m for m in models if m.supports_multivariate]


def get_univariate_models(models: list[BaseForecaster]) -> list[BaseForecaster]:
    """Filter models that are univariate only.

    Args:
        models: List of forecaster instances.

    Returns:
        List of univariate forecasters.
    """
    return [m for m in models if not m.supports_multivariate]
