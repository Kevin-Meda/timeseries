"""Main pipeline orchestrator for the forecasting platform."""

import json
from pathlib import Path

import pandas as pd
import yaml

from forecast.utils import setup_logger, get_logger
from forecast.data import load_excel, split_series
from forecast.preprocessing import classify_series, preprocess_series
from forecast.preprocessing.classifier import classify_all_series
from forecast.preprocessing.cleaner import preprocess_all_series
from forecast.models import SARIMAForecaster, ChronosForecaster, HoltWintersForecaster
from forecast.models.chronos import is_chronos_available
from forecast.evaluation import evaluate_all, create_ensemble
from forecast.output import write_model_results, write_ensemble_results
from forecast.plotting import plot_preprocessing, plot_evaluation, plot_forecast
from forecast.plotting.preprocessing_plots import plot_all_preprocessing
from forecast.plotting.evaluation_plots import plot_all_evaluations
from forecast.plotting.forecast_plots import plot_all_forecasts


def load_config(config_dir: str) -> dict:
    """Load all configuration files from directory.

    Args:
        config_dir: Path to configuration directory.

    Returns:
        Combined configuration dictionary.
    """
    config_path = Path(config_dir)
    config = {}

    config_files = [
        "data_input.yaml",
        "preprocessing.yaml",
        "models.yaml",
        "pipeline.yaml",
        "plots.yaml",
        "output.yaml",
    ]

    for filename in config_files:
        filepath = config_path / filename
        if filepath.exists():
            with open(filepath, "r") as f:
                file_config = yaml.safe_load(f) or {}
                key = filename.replace(".yaml", "")
                config[key] = file_config

    return config


def run_pipeline(
    config_dir: str = "configs",
    input_override: str | None = None,
    output_dir_override: str | None = None,
    log_level: str = "INFO",
) -> dict:
    """Run the complete forecasting pipeline.

    Args:
        config_dir: Path to configuration directory.
        input_override: Optional override for input Excel path.
        output_dir_override: Optional override for output directory.
        log_level: Logging level.

    Returns:
        Dictionary containing all results.
    """
    logger = setup_logger(level=log_level)
    logger.info("Starting forecasting pipeline")

    config = load_config(config_dir)
    logger.info(f"Loaded configuration from: {config_dir}")

    data_config = config.get("data_input", {})
    preprocess_config = config.get("preprocessing", {})
    models_config = config.get("models", {})
    pipeline_config = config.get("pipeline", {})
    plots_config = config.get("plots", {})
    output_config = config.get("output", {})

    excel_path = input_override or data_config.get("excel_path", "data/input/demand.xlsx")
    results_dir = output_dir_override or output_config.get("results_dir", "output/results")
    models_dir = output_config.get("models_dir", "output/models")
    plots_dir = output_config.get("plots_dir", "output/plots")

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from: {excel_path}")
    try:
        raw_data = load_excel(excel_path, data_config)
    except FileNotFoundError:
        logger.error(f"Input file not found: {excel_path}")
        return {"error": f"Input file not found: {excel_path}"}

    classification_config = preprocess_config.get("classification", {})
    lookback = classification_config.get("lookback_months", 24)
    min_pct = classification_config.get("min_nonzero_pct", 0.5)

    forecastable, non_forecastable = classify_all_series(raw_data, lookback, min_pct)
    logger.info(f"Classified {len(forecastable)} forecastable series")

    if not forecastable:
        logger.warning("No forecastable series found")
        return {"error": "No forecastable series found"}

    cleaned_data, change_logs = preprocess_all_series(forecastable, preprocess_config)

    val_months = pipeline_config.get("validation_months", 12)
    test_months = pipeline_config.get("test_months", 12)
    window_months = preprocess_config.get("data_window_months", 60)
    horizon = pipeline_config.get("forecast_horizon", 12)
    mape_threshold = pipeline_config.get("ensemble_mape_threshold", 0.5)

    splits_dict = {}
    for name, series in cleaned_data.items():
        splits_dict[name] = split_series(series, val_months, test_months, window_months)

    if plots_config.get("save_preprocessing", True):
        plot_all_preprocessing(
            original_dict=forecastable,
            cleaned_dict=cleaned_data,
            change_logs=change_logs,
            splits_dict=splits_dict,
            output_dir=f"{plots_dir}/preprocessing",
            dpi=plots_config.get("figure_dpi", 150),
        )

    def create_models():
        """Create fresh model instances."""
        models = []
        if models_config.get("sarima", {}).get("enabled", True):
            sarima_config = models_config.get("sarima", {})
            models.append(
                SARIMAForecaster(
                    use_optuna=sarima_config.get("use_optuna", False),
                    optuna_trials=sarima_config.get("optuna_trials", 50),
                )
            )

        if models_config.get("chronos", {}).get("enabled", True):
            if is_chronos_available():
                models.append(ChronosForecaster())
            else:
                logger.warning("Chronos enabled but dependencies not available")

        if models_config.get("holt_winters", {}).get("enabled", False):
            models.append(HoltWintersForecaster())

        return models

    # Check if any models are enabled
    test_models = create_models()
    if not test_models:
        logger.error("No models enabled for training")
        return {"error": "No models enabled"}

    logger.info(f"Training models: {[m.name for m in test_models]}")

    all_results = {}
    all_test_predictions = {}
    all_test_actuals = {}
    all_forecasts = {}
    ensemble_results = {}

    for category_name, splits in splits_dict.items():
        logger.info(f"Processing category: {category_name}")

        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        category_results = {}
        category_test_predictions = {}
        category_forecasts = {}

        # Create fresh model instances for each category
        models_to_train = create_models()

        for model in models_to_train:
            logger.debug(f"Training {model.name} for {category_name}")

            try:
                model.fit(train, val)

                if not model.is_fitted:
                    logger.warning(f"{model.name} failed to fit for {category_name}")
                    continue

                test_pred = model.predict(len(test))

                if hasattr(test_pred, "index") and hasattr(test, "index"):
                    test_pred.index = test.index

                metrics = evaluate_all(test.values, test_pred.values)
                category_results[model.name] = metrics
                category_test_predictions[model.name] = test_pred

                logger.info(
                    f"{model.name} - {category_name}: "
                    f"RMSE={metrics['rmse']:.2f}, "
                    f"MAPE={metrics['mape']:.2%}, "
                    f"MAE={metrics['mae']:.2f}"
                )

                combined_train = pd.concat([train, val, test])
                model.fit(combined_train, None)

                if model.is_fitted:
                    future_forecast = model.predict(horizon)
                    category_forecasts[model.name] = future_forecast

                    params_path = Path(models_dir) / f"{category_name}_{model.name}_params.json"
                    safe_path = str(params_path).replace(" ", "_")
                    with open(safe_path, "w") as f:
                        json.dump(model.get_params(), f, indent=2)

            except Exception as e:
                logger.error(f"Error training {model.name} for {category_name}: {e}")
                continue

        if category_results:
            all_results[category_name] = category_results
            all_test_predictions[category_name] = category_test_predictions
            all_test_actuals[category_name] = test
            all_forecasts[category_name] = category_forecasts

            if len(category_forecasts) > 0:
                ensemble, weights, models_used = create_ensemble(
                    category_forecasts, category_results, mape_threshold
                )

                if len(ensemble) > 0:
                    ensemble_results[category_name] = {
                        "forecast": ensemble,
                        "mape": category_results.get(models_used[0], {}).get("mape", 0)
                        if models_used else 0,
                        "models_used": models_used,
                        "weights": weights,
                    }

    if plots_config.get("save_evaluation", True):
        plot_all_evaluations(
            actuals=all_test_actuals,
            predictions=all_test_predictions,
            metrics=all_results,
            output_dir=f"{plots_dir}/evaluation",
            dpi=plots_config.get("figure_dpi", 150),
        )

    if plots_config.get("save_forecast", True):
        histories = {name: splits_dict[name]["full"] for name in all_forecasts.keys()}
        ensemble_forecasts = {name: info["forecast"] for name, info in ensemble_results.items()}

        plot_all_forecasts(
            histories=histories,
            forecasts=all_forecasts,
            ensemble_forecasts=ensemble_forecasts,
            output_dir=f"{plots_dir}/forecast",
            dpi=plots_config.get("figure_dpi", 150),
        )

    write_model_results(all_results, f"{results_dir}/model_results.xlsx")

    write_ensemble_results(ensemble_results, f"{results_dir}/ensemble_forecasts.xlsx")

    logger.info("Pipeline completed successfully")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Model parameters saved to: {models_dir}")
    logger.info(f"Plots saved to: {plots_dir}")

    return {
        "model_results": all_results,
        "ensemble_results": ensemble_results,
        "forecasts": all_forecasts,
        "categories_processed": len(all_results),
    }
