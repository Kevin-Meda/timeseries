"""Main pipeline orchestrator for the forecasting platform."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
import yaml

from forecast.utils import setup_logger, get_logger
from forecast.utils.project_manager import ProjectManager
from forecast.data import load_excel, split_series
from forecast.data.splitter import split_with_features
from forecast.preprocessing import classify_series, preprocess_series
from forecast.preprocessing.classifier import classify_all_series
from forecast.preprocessing.cleaner import preprocess_all_series
from forecast.models import create_models, get_available_models
from forecast.models.base import BaseForecaster
from forecast.evaluation import evaluate_all, create_ensemble
from forecast.output import (
    write_model_results,
    write_ensemble_results,
    write_run_summary,
    build_product_result,
    build_model_result,
)
from forecast.plotting import plot_preprocessing, plot_model_comparison, plot_forecast
from forecast.plotting.preprocessing_plots import plot_all_preprocessing
from forecast.plotting.evaluation_plots import plot_all_evaluations
from forecast.plotting.forecast_plots import plot_all_forecasts
from forecast.tuning.param_store import (
    ParamStore,
    should_optimize,
    get_model_config_key,
)
from forecast.features.feature_store import FeatureStoreLoader
from forecast.features.encoders import FoldAwareEncoder
from forecast.features.scalers import FoldAwareScaler


def get_timestamp() -> str:
    """Get current timestamp for output filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def load_config(config_dir: str) -> dict:
    """Load all configuration files from directory."""
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


def _extract_target(data: pd.Series | pd.DataFrame) -> pd.Series:
    """Extract target series from data (Series or DataFrame).

    Args:
        data: Input data.

    Returns:
        Target series.
    """
    if isinstance(data, pd.DataFrame):
        if "demand" in data.columns:
            return data["demand"]
        return data.iloc[:, 0]
    return data


def _get_exog_features(data: pd.DataFrame, target_col: str = "demand") -> pd.DataFrame | None:
    """Extract exogenous features from DataFrame.

    Args:
        data: DataFrame with target and features.
        target_col: Name of target column to exclude.

    Returns:
        DataFrame with features only, or None if no features.
    """
    if not isinstance(data, pd.DataFrame):
        return None

    feature_cols = [col for col in data.columns if col != target_col]
    if not feature_cols:
        return None

    return data[feature_cols]


def run_pipeline(
    config_dir: str = "configs",
    input_override: str | None = None,
    output_dir_override: str | None = None,
    log_level: str = "INFO",
    project_name: str = "default",
) -> dict:
    """Run the complete forecasting pipeline.

    Args:
        config_dir: Path to configuration directory.
        input_override: Optional override for input file path.
        output_dir_override: Optional override for output directory.
        log_level: Logging level.
        project_name: Project name for organizing outputs.

    Returns:
        Dictionary with pipeline results.
    """
    # Setup logger with project name
    logger = setup_logger(level=log_level, project_name=project_name)
    logger.info(f"Starting forecasting pipeline - Project: {project_name}")

    # Initialize project manager
    pm = ProjectManager(
        project_name=project_name,
        base_output_dir=output_dir_override or "output",
    )
    timestamp = pm.timestamp
    logger.info(f"Run timestamp: {timestamp}")

    # Load configuration
    config = load_config(config_dir)
    logger.info(f"Loaded configuration from: {config_dir}")

    data_config = config.get("data_input", {})
    preprocess_config = config.get("preprocessing", {})
    models_config = config.get("models", {})
    pipeline_config = config.get("pipeline", {})
    plots_config = config.get("plots", {})
    output_config = config.get("output", {})

    # Data paths
    excel_path = input_override or data_config.get("excel_path", "data/input/demand.xlsx")
    feature_store_path = data_config.get("feature_store_path")
    categorical_features = data_config.get("categorical_features", [])

    # Get directories from project manager
    results_dir = pm.results_dir
    models_dir = pm.models_dir
    plots_dir = pm.plots_dir

    # Pipeline settings
    product_mode = pipeline_config.get("product_mode", "per_product")
    encode_product_id = pipeline_config.get("encode_product_id", True)
    val_months = pipeline_config.get("validation_months", 12)
    test_months = pipeline_config.get("test_months", 12)
    window_months = preprocess_config.get("data_window_months", 60)
    horizon = pipeline_config.get("forecast_horizon", 12)
    mape_threshold = pipeline_config.get("ensemble_mape_threshold", 0.5)

    # Scaler settings
    scaler_config = preprocess_config.get("scaler", {})
    use_scaler = scaler_config.get("enabled", False)
    scaler_type = scaler_config.get("type", "robust")

    # Load data
    logger.info(f"Loading data from: {excel_path}")
    try:
        raw_data = load_excel(excel_path, data_config)
    except FileNotFoundError:
        logger.error(f"Input file not found: {excel_path}")
        return {"error": f"Input file not found: {excel_path}"}

    # Load feature store if configured
    feature_store = None
    features_by_product: dict[str, pd.DataFrame] = {}
    if feature_store_path:
        logger.info(f"Loading feature store from: {feature_store_path}")
        try:
            feature_store = FeatureStoreLoader(feature_store_path)
            feature_store.load()
        except FileNotFoundError:
            logger.warning(f"Feature store not found: {feature_store_path}")
            feature_store = None

    # Classify series
    classification_config = preprocess_config.get("classification", {})
    lookback = classification_config.get("lookback_months", 24)
    min_pct = classification_config.get("min_nonzero_pct", 0.5)

    forecastable, non_forecastable = classify_all_series(raw_data, lookback, min_pct)
    logger.info(f"Classified {len(forecastable)} forecastable series")

    if non_forecastable:
        logger.info(f"Skipped {len(non_forecastable)} non-forecastable series: {list(non_forecastable.keys())}")

    if not forecastable:
        logger.warning("No forecastable series found")
        return {"error": "No forecastable series found"}

    # Window the data
    windowed_data = {}
    for name, series in forecastable.items():
        if len(series) > window_months:
            windowed_data[name] = series.iloc[-window_months:]
        else:
            windowed_data[name] = series

    # Clean the windowed data
    cleaned_data, change_logs = preprocess_all_series(windowed_data, preprocess_config)

    # Load features for each product
    if feature_store:
        for product_name, demand_series in cleaned_data.items():
            features_df = feature_store.get_product_features(product_name, demand_series)
            if len(features_df.columns) > 1:  # Has features beyond demand
                features_by_product[product_name] = features_df
                logger.debug(f"Loaded {len(features_df.columns) - 1} features for {product_name}")

    # Split data
    splits_dict = {}
    for name, series in cleaned_data.items():
        if name in features_by_product:
            # Use features-aware splitting
            splits_dict[name] = split_with_features(
                series, features_by_product[name], val_months, test_months, window_months
            )
        else:
            splits_dict[name] = split_series(series, val_months, test_months, window_months)

    # Plot preprocessing
    if plots_config.get("save_preprocessing", True):
        plot_all_preprocessing(
            original_dict=windowed_data,
            cleaned_dict=cleaned_data,
            change_logs=change_logs,
            splits_dict=splits_dict,
            output_dir=str(plots_dir / "preprocessing"),
            dpi=plots_config.get("figure_dpi", 150),
        )

    # Create models from config
    models_list = create_models(models_config, product_mode)
    if not models_list:
        logger.error("No models enabled for training")
        return {"error": "No models enabled"}

    logger.info(f"Training models: {[m.name for m in models_list]}")

    # Initialize result containers
    all_results: dict[str, dict[str, dict]] = {}
    all_test_predictions: dict[str, dict[str, pd.Series]] = {}
    all_test_actuals: dict[str, pd.Series] = {}
    all_forecasts: dict[str, dict[str, pd.Series]] = {}
    ensemble_results: dict[str, dict] = {}
    products_json_data: dict[str, dict] = {}
    all_model_weights: dict[str, dict[str, float]] = {}  # Track weights for all models

    # Initialize param store
    param_store = ParamStore(pm)

    # Process each category/product
    for category_name, splits in splits_dict.items():
        logger.info(f"Processing category: {category_name}")

        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        # Extract targets for evaluation
        train_target = _extract_target(train)
        val_target = _extract_target(val)
        test_target = _extract_target(test)

        # Extract exogenous features
        exog_train = _get_exog_features(train) if isinstance(train, pd.DataFrame) else None
        exog_val = _get_exog_features(val) if isinstance(val, pd.DataFrame) else None
        exog_test = _get_exog_features(test) if isinstance(test, pd.DataFrame) else None

        # Apply encoding and scaling if features present
        encoder = None
        scaler = None

        if exog_train is not None and len(exog_train.columns) > 0:
            # Fit encoder on training data only (data leakage prevention)
            encoder = FoldAwareEncoder()
            encoder.fit(exog_train, categorical_features)
            exog_train = encoder.transform(exog_train)
            if exog_val is not None:
                exog_val = encoder.transform(exog_val)
            if exog_test is not None:
                exog_test = encoder.transform(exog_test)

            # Fit scaler on training data only
            if use_scaler:
                scaler = FoldAwareScaler(scaler_type)
                scaler.fit(exog_train, exclude_columns=["demand"])
                exog_train = scaler.transform(exog_train)
                if exog_val is not None:
                    exog_val = scaler.transform(exog_val)
                if exog_test is not None:
                    exog_test = scaler.transform(exog_test)

        category_results: dict[str, dict] = {}
        category_test_predictions: dict[str, pd.Series] = {}
        category_forecasts: dict[str, pd.Series] = {}
        category_model_json: dict[str, dict] = {}

        # Train each model
        for model in models_list:
            # Create fresh model instance for this category
            model_instance = _create_fresh_model(model, models_config)
            model_name = model_instance.name

            logger.debug(f"Training {model_name} for {category_name}")

            try:
                # Check if we should optimize or load params (per-model)
                do_optimize, loaded_params = should_optimize(
                    models_config, pm, model_name, category_name
                )

                if not do_optimize and loaded_params:
                    # Load existing or default parameters
                    model_instance.load_params(loaded_params)
                    logger.info(f"Loaded params for {model_name}/{category_name}")

                # Fit model
                if model_instance.supports_multivariate and exog_train is not None:
                    model_instance.fit(train_target, val_target, exog_train, exog_val)
                else:
                    model_instance.fit(train_target, val_target)

                if not model_instance.is_fitted:
                    logger.warning(f"{model_name} failed to fit for {category_name}")
                    continue

                # Predict on test set
                if model_instance.supports_multivariate and exog_test is not None:
                    test_pred = model_instance.predict(len(test_target), exog_test)
                else:
                    test_pred = model_instance.predict(len(test_target))

                # Align indices
                if hasattr(test_pred, "index") and hasattr(test_target, "index"):
                    test_pred.index = test_target.index

                # Evaluate
                metrics = evaluate_all(test_target.values, test_pred.values)
                category_results[model_name] = metrics
                category_test_predictions[model_name] = test_pred

                logger.info(
                    f"{model_name} - {category_name}: "
                    f"RMSE={metrics['rmse']:.2f}, "
                    f"MAPE={metrics['mape']:.2%}, "
                    f"MAE={metrics['mae']:.2f}"
                )

                # Re-train on full data and generate forecasts
                combined_target = pd.concat([train_target, val_target, test_target])
                combined_exog = None
                if exog_train is not None:
                    combined_exog = pd.concat([exog_train, exog_val, exog_test])

                if model_instance.supports_multivariate and combined_exog is not None:
                    model_instance.fit(combined_target, None, combined_exog, None)
                else:
                    model_instance.fit(combined_target, None)

                if model_instance.is_fitted:
                    # Generate future forecasts
                    exog_future = None
                    if feature_store and model_instance.supports_multivariate:
                        # Extrapolate features for forecast period
                        last_date = combined_target.index[-1]
                        future_index = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=horizon,
                            freq="MS",
                        )
                        if combined_exog is not None:
                            exog_future = feature_store.extrapolate_features(
                                combined_exog, future_index
                            )
                            if encoder:
                                exog_future = encoder.transform(exog_future)
                            if scaler:
                                exog_future = scaler.transform(exog_future)

                    if model_instance.supports_multivariate and exog_future is not None:
                        future_forecast = model_instance.predict(horizon, exog_future)
                    else:
                        future_forecast = model_instance.predict(horizon)

                    category_forecasts[model_name] = future_forecast

                    # Save parameters
                    params = model_instance.get_params()
                    param_store.save_params(model_name, category_name, params)

                    # Build JSON result
                    feature_importance = None
                    if hasattr(model_instance, "get_feature_importance"):
                        feature_importance = model_instance.get_feature_importance()

                    optuna_trials = params.get("optuna_trials_used", 0)

                    category_model_json[model_name] = build_model_result(
                        metrics=metrics,
                        params=params,
                        retrained=do_optimize,
                        optuna_trials_used=optuna_trials,
                        feature_importance=feature_importance,
                    )

            except Exception as e:
                logger.error(f"Error training {model_name} for {category_name}: {e}")
                continue

        # Store results for this category
        if category_results:
            all_results[category_name] = category_results
            all_test_predictions[category_name] = category_test_predictions
            all_test_actuals[category_name] = test_target
            all_forecasts[category_name] = category_forecasts
            products_json_data[category_name] = build_product_result(
                category_name, category_model_json
            )

            # Create ensemble and track all model weights
            if len(category_forecasts) > 0:
                ensemble, weights, models_used = create_ensemble(
                    category_forecasts, category_results, mape_threshold
                )

                # Build complete weights dict (including 0% for excluded models)
                complete_weights = {}
                for model_name in category_forecasts.keys():
                    complete_weights[model_name] = weights.get(model_name, 0.0)

                all_model_weights[category_name] = complete_weights

                if len(ensemble) > 0:
                    ensemble_results[category_name] = {
                        "forecast": ensemble,
                        "mape": category_results.get(models_used[0], {}).get("mape", 0)
                        if models_used else 0,
                        "models_used": models_used,
                        "weights": weights,
                    }

    # Generate plots
    if plots_config.get("save_evaluation", True):
        plot_all_evaluations(
            actuals=all_test_actuals,
            predictions=all_test_predictions,
            metrics=all_results,
            output_dir=str(plots_dir / "evaluation"),
            dpi=plots_config.get("figure_dpi", 150),
        )

    if plots_config.get("save_forecast", True):
        # Extract histories (demand only for plotting)
        histories = {}
        for name in all_forecasts.keys():
            full_data = splits_dict[name]["full"]
            if isinstance(full_data, pd.DataFrame):
                histories[name] = full_data["demand"] if "demand" in full_data.columns else full_data.iloc[:, 0]
            else:
                histories[name] = full_data

        ensemble_forecasts = {name: info["forecast"] for name, info in ensemble_results.items()}

        plot_all_forecasts(
            histories=histories,
            forecasts=all_forecasts,
            ensemble_forecasts=ensemble_forecasts,
            ensemble_weights=all_model_weights,
            output_dir=str(plots_dir / "forecast"),
            dpi=plots_config.get("figure_dpi", 150),
        )

    # Write outputs
    write_model_results(
        all_results,
        all_test_actuals,
        all_test_predictions,
        str(results_dir / "model_results.xlsx")
    )

    write_ensemble_results(ensemble_results, str(results_dir / "ensemble_forecasts.xlsx"))

    # Write consolidated JSON summary
    write_run_summary(
        project_name=project_name,
        timestamp=timestamp,
        results_dir=results_dir,
        products_data=products_json_data,
        run_config={
            "models": models_config,
            "pipeline": pipeline_config,
            "preprocessing": preprocess_config,
        },
    )

    logger.info("Pipeline completed successfully")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Model parameters saved to: {models_dir}")
    logger.info(f"Plots saved to: {plots_dir}")

    return {
        "model_results": all_results,
        "ensemble_results": ensemble_results,
        "forecasts": all_forecasts,
        "categories_processed": len(all_results),
        "timestamp": timestamp,
        "project_name": project_name,
    }


def _create_fresh_model(template: BaseForecaster, config: dict) -> BaseForecaster:
    """Create a fresh model instance based on a template.

    Uses per-model optimize_params flag instead of global retrain.

    Args:
        template: Model instance to copy settings from.
        config: Models configuration dictionary.

    Returns:
        New model instance.
    """
    from forecast.models import (
        SARIMAForecaster,
        HoltWintersForecaster,
        ChronosForecaster,
        XGBoostForecaster,
        ProphetForecaster,
    )
    from forecast.models.xgboost import is_xgboost_available
    from forecast.models.prophet import is_prophet_available

    if template.name == "SARIMA":
        sarima_config = config.get("sarima", {})
        defaults = sarima_config.get("defaults", {})
        optimize = sarima_config.get("optimize_params", False)
        return SARIMAForecaster(
            use_optuna=optimize,
            optuna_trials=sarima_config.get("optuna_trials", 50),
        )

    elif template.name == "HoltWinters":
        hw_config = config.get("holt_winters", {})
        defaults = hw_config.get("defaults", {})
        return HoltWintersForecaster(
            trend=defaults.get("trend", "add"),
            seasonal=defaults.get("seasonal", "add"),
            seasonal_periods=defaults.get("seasonal_periods", 12),
        )

    elif template.name == "Chronos":
        chronos_config = config.get("chronos", {})
        defaults = chronos_config.get("defaults", {})
        return ChronosForecaster(
            model_name=defaults.get("model_name", "amazon/chronos-t5-small"),
        )

    elif template.name == "XGBoost" and is_xgboost_available():
        xgb_config = config.get("xgboost", {})
        defaults = xgb_config.get("defaults", {})
        optimize = xgb_config.get("optimize_params", True)
        return XGBoostForecaster(
            use_optuna=optimize,
            optuna_trials=xgb_config.get("optuna_trials", 100),
            n_estimators=defaults.get("n_estimators", 100),
            max_depth=defaults.get("max_depth", 6),
            learning_rate=defaults.get("learning_rate", 0.1),
        )

    elif template.name == "Prophet" and is_prophet_available():
        prophet_config = config.get("prophet", {})
        defaults = prophet_config.get("defaults", {})
        optimize = prophet_config.get("optimize_params", True)
        return ProphetForecaster(
            use_optuna=optimize,
            optuna_trials=prophet_config.get("optuna_trials", 50),
            changepoint_prior_scale=defaults.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=defaults.get("seasonality_prior_scale", 10.0),
            seasonality_mode=defaults.get("seasonality_mode", "additive"),
        )

    # Fallback: return the template itself (not ideal but prevents crashes)
    return template
