"""Common functions shared across all pipeline types."""

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
from forecast.preprocessing.classifier import classify_all_series
from forecast.preprocessing.cleaner import preprocess_all_series
from forecast.models.base import BaseForecaster
from forecast.evaluation import (
    evaluate_all,
    create_ensemble,
    compute_feature_importance,
)
from forecast.output import (
    write_model_results,
    write_ensemble_results,
    write_run_summary,
    build_product_result,
    build_model_result,
)
from forecast.plotting.preprocessing_plots import plot_all_preprocessing
from forecast.plotting.evaluation_plots import plot_all_evaluations
from forecast.plotting.forecast_plots import plot_all_forecasts
from forecast.tuning.param_store import ParamStore, should_optimize
from forecast.features.feature_store import FeatureStoreLoader
from forecast.features.encoders import FoldAwareEncoder
from forecast.features.scalers import FoldAwareScaler


def load_config(config_dir: str) -> dict:
    """Load all configuration files from directory.

    Args:
        config_dir: Path to configuration directory.

    Returns:
        Dictionary containing all configuration sections.
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


def validate_univariate_config(config: dict) -> None:
    """Validate configuration for univariate pipeline.

    Univariate pipelines do not require feature_store_path.

    Args:
        config: Full configuration dictionary.

    Raises:
        ValueError: If required configuration is missing.
    """
    data_config = config.get("data_input", {})
    excel_path = data_config.get("excel_path")

    if not excel_path:
        raise ValueError("Missing required config: data_input.excel_path")


def validate_multivariate_config(config: dict) -> None:
    """Validate configuration for multivariate pipeline.

    Multivariate pipelines require feature_store_path.

    Args:
        config: Full configuration dictionary.

    Raises:
        ValueError: If required configuration is missing.
    """
    data_config = config.get("data_input", {})
    excel_path = data_config.get("excel_path")
    feature_store_path = data_config.get("feature_store_path")

    if not excel_path:
        raise ValueError("Missing required config: data_input.excel_path")

    if not feature_store_path:
        raise ValueError(
            "Multivariate pipeline requires feature_store_path in data_input.yaml. "
            "Set feature_store_path to the path of your features Excel file."
        )


def setup_pipeline_run(
    project_name: str,
    log_level: str,
    output_dir_override: str | None,
    pipeline_type: str = "default",
) -> tuple[Any, ProjectManager, str]:
    """Initialize logger and project manager for pipeline run.

    Args:
        project_name: Name of the project.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        output_dir_override: Optional override for output directory.
        pipeline_type: Type of pipeline (univariate, multivariate_single, multivariate_all).

    Returns:
        Tuple of (logger, ProjectManager, timestamp).
    """
    logger = setup_logger(level=log_level, project_name=project_name)
    logger.info(f"Starting {pipeline_type} pipeline - Project: {project_name}")

    pm = ProjectManager(
        project_name=project_name,
        base_output_dir=output_dir_override or "output",
        pipeline_type=pipeline_type,
    )
    timestamp = pm.timestamp
    logger.info(f"Run timestamp: {timestamp}")

    return logger, pm, timestamp


def load_and_preprocess_data(
    config: dict,
    input_override: str | None = None,
) -> tuple[dict[str, pd.Series], dict[str, pd.Series], dict[str, list]]:
    """Load and preprocess data from Excel file.

    Args:
        config: Full configuration dictionary.
        input_override: Optional override for input file path.

    Returns:
        Tuple of (cleaned_data, windowed_data, change_logs).
    """
    logger = get_logger()

    data_config = config.get("data_input", {})
    preprocess_config = config.get("preprocessing", {})

    excel_path = input_override or data_config.get("excel_path", "data/input/demand.xlsx")
    window_months = preprocess_config.get("data_window_months", 60)

    # Load data
    logger.info(f"Loading data from: {excel_path}")
    raw_data = load_excel(excel_path, data_config)

    # Classify series
    classification_config = preprocess_config.get("classification", {})
    lookback = classification_config.get("lookback_months", 24)
    min_pct = classification_config.get("min_nonzero_pct", 0.5)

    forecastable, non_forecastable = classify_all_series(raw_data, lookback, min_pct)
    logger.info(f"Classified {len(forecastable)} forecastable series")

    if non_forecastable:
        logger.info(
            f"Skipped {len(non_forecastable)} non-forecastable series: "
            f"{list(non_forecastable.keys())}"
        )

    if not forecastable:
        raise ValueError("No forecastable series found")

    # Window the data
    windowed_data = {}
    for name, series in forecastable.items():
        if len(series) > window_months:
            windowed_data[name] = series.iloc[-window_months:]
        else:
            windowed_data[name] = series

    # Clean the windowed data
    cleaned_data, change_logs = preprocess_all_series(windowed_data, preprocess_config)

    return cleaned_data, windowed_data, change_logs


def normalize_product_name(name: str) -> str:
    """Normalize product name for feature store lookup.

    Replaces spaces with underscores to match feature store column naming.

    Args:
        name: Product name (e.g., "Product A").

    Returns:
        Normalized name (e.g., "Product_A").
    """
    return name.replace(" ", "_")


def load_feature_store(
    config: dict,
    cleaned_data: dict[str, pd.Series],
) -> tuple[FeatureStoreLoader | None, dict[str, pd.DataFrame]]:
    """Load feature store and get features for each product.

    Args:
        config: Full configuration dictionary.
        cleaned_data: Dictionary of cleaned demand series by product.

    Returns:
        Tuple of (feature_store, features_by_product).
    """
    logger = get_logger()

    data_config = config.get("data_input", {})
    feature_store_path = data_config.get("feature_store_path")
    selected_features = data_config.get("selected_features", {})

    if not feature_store_path:
        return None, {}

    logger.info(f"Loading feature store from: {feature_store_path}")
    try:
        feature_store = FeatureStoreLoader(feature_store_path)
        feature_store.load()
    except FileNotFoundError:
        logger.warning(f"Feature store not found: {feature_store_path}")
        return None, {}

    features_by_product: dict[str, pd.DataFrame] = {}
    for product_name, demand_series in cleaned_data.items():
        # Normalize product name for feature store lookup (spaces -> underscores)
        normalized_name = normalize_product_name(product_name)
        features_df = feature_store.get_product_features(normalized_name, demand_series)

        # Filter to selected features if specified
        if product_name in selected_features and selected_features[product_name]:
            available_cols = set(features_df.columns) - {"demand"}
            selected_cols = set(selected_features[product_name])
            cols_to_keep = ["demand"] + [c for c in selected_cols if c in available_cols]
            features_df = features_df[cols_to_keep]

        if len(features_df.columns) > 1:  # Has features beyond demand
            features_by_product[product_name] = features_df
            logger.debug(
                f"Loaded {len(features_df.columns) - 1} features for {product_name}"
            )

    return feature_store, features_by_product


def build_cross_product_features(
    cleaned_data: dict[str, pd.Series],
    feature_store_features: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Build features including other products' demand as features.

    For each target product, adds all OTHER products as demand features.

    Args:
        cleaned_data: Dictionary mapping product names to demand series.
        feature_store_features: Dictionary mapping product names to feature DataFrames.

    Returns:
        Dictionary mapping product names to DataFrames with cross-product features.
    """
    logger = get_logger()
    result = {}
    products = list(cleaned_data.keys())

    for target in products:
        # Start with feature store features or create empty DataFrame
        if target in feature_store_features:
            features = feature_store_features[target].copy()
        else:
            features = pd.DataFrame(index=cleaned_data[target].index)
            features["demand"] = cleaned_data[target]

        # Add other products as demand features
        for other in products:
            if other != target:
                other_series = cleaned_data[other]
                # Align indices
                aligned_series = other_series.reindex(features.index)
                # Normalize name for consistent feature naming
                normalized_other = normalize_product_name(other)
                features[f"{normalized_other}_demand"] = aligned_series

        result[target] = features
        logger.debug(
            f"Built cross-product features for {target}: "
            f"{len(features.columns) - 1} features"
        )

    return result


def split_data(
    cleaned_data: dict[str, pd.Series],
    features_by_product: dict[str, pd.DataFrame],
    config: dict,
) -> dict[str, dict]:
    """Split data into train/val/test sets.

    Args:
        cleaned_data: Dictionary of cleaned demand series.
        features_by_product: Dictionary of feature DataFrames by product.
        config: Full configuration dictionary.

    Returns:
        Dictionary mapping product names to split dictionaries.
    """
    pipeline_config = config.get("pipeline", {})
    preprocess_config = config.get("preprocessing", {})

    val_months = pipeline_config.get("validation_months", 12)
    test_months = pipeline_config.get("test_months", 12)
    window_months = preprocess_config.get("data_window_months", 60)

    splits_dict = {}
    for name, series in cleaned_data.items():
        if name in features_by_product:
            splits_dict[name] = split_with_features(
                series, features_by_product[name], val_months, test_months, window_months
            )
        else:
            splits_dict[name] = split_series(series, val_months, test_months, window_months)

    return splits_dict


def extract_target(data: pd.Series | pd.DataFrame) -> pd.Series:
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


def get_exog_features(data: pd.DataFrame, target_col: str = "demand") -> pd.DataFrame | None:
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


def encode_and_scale_features(
    exog_train: pd.DataFrame | None,
    exog_val: pd.DataFrame | None,
    exog_test: pd.DataFrame | None,
    categorical_features: list[str],
    use_scaler: bool,
    scaler_type: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, Any, Any]:
    """Apply encoding and scaling to exogenous features.

    Args:
        exog_train: Training features.
        exog_val: Validation features.
        exog_test: Test features.
        categorical_features: List of categorical feature names.
        use_scaler: Whether to apply scaling.
        scaler_type: Type of scaler to use.

    Returns:
        Tuple of (exog_train, exog_val, exog_test, encoder, scaler).
    """
    encoder = None
    scaler = None

    if exog_train is None or len(exog_train.columns) == 0:
        return exog_train, exog_val, exog_test, encoder, scaler

    # Fit encoder on training data only
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

    return exog_train, exog_val, exog_test, encoder, scaler


def train_single_model(
    model: BaseForecaster,
    train_target: pd.Series,
    val_target: pd.Series,
    test_target: pd.Series,
    exog_train: pd.DataFrame | None,
    exog_val: pd.DataFrame | None,
    exog_test: pd.DataFrame | None,
    category_name: str,
    models_config: dict,
    pm: ProjectManager,
    param_store: ParamStore,
    feature_store: FeatureStoreLoader | None,
    horizon: int,
    encoder: Any,
    scaler: Any,
    compute_fi: bool = False,
) -> dict[str, Any] | None:
    """Train a single model and generate predictions.

    Args:
        model: Model instance to train.
        train_target: Training target series.
        val_target: Validation target series.
        test_target: Test target series.
        exog_train: Training exogenous features.
        exog_val: Validation exogenous features.
        exog_test: Test exogenous features.
        category_name: Name of the category/product.
        models_config: Models configuration.
        pm: Project manager.
        param_store: Parameter store.
        feature_store: Feature store loader.
        horizon: Forecast horizon.
        encoder: Fitted encoder for features.
        scaler: Fitted scaler for features.
        compute_fi: Whether to compute feature importance.

    Returns:
        Dictionary with model results or None if training failed.
    """
    logger = get_logger()
    model_name = model.name

    logger.debug(f"Training {model_name} for {category_name}")

    try:
        # Check if we should optimize or load params
        do_optimize, loaded_params = should_optimize(
            models_config, pm, model_name, category_name
        )

        if not do_optimize and loaded_params:
            model.load_params(loaded_params)
            logger.info(f"Loaded params for {model_name}/{category_name}")

        # Fit model
        if model.supports_multivariate and exog_train is not None:
            model.fit(train_target, val_target, exog_train, exog_val)
        else:
            model.fit(train_target, val_target)

        if not model.is_fitted:
            logger.warning(f"{model_name} failed to fit for {category_name}")
            return None

        # Predict on test set
        if model.supports_multivariate and exog_test is not None:
            test_pred = model.predict(len(test_target), exog_test)
        else:
            test_pred = model.predict(len(test_target))

        # Align indices
        if hasattr(test_pred, "index") and hasattr(test_target, "index"):
            test_pred.index = test_target.index

        # Evaluate
        metrics = evaluate_all(test_target.values, test_pred.values)

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

        if model.supports_multivariate and combined_exog is not None:
            model.fit(combined_target, None, combined_exog, None)
        else:
            model.fit(combined_target, None)

        future_forecast = None
        feature_importance_result = None

        if model.is_fitted:
            # Generate future forecasts
            exog_future = None
            if feature_store and model.supports_multivariate and combined_exog is not None:
                last_date = combined_target.index[-1]
                future_index = pd.date_range(
                    start=last_date + pd.DateOffset(months=1),
                    periods=horizon,
                    freq="MS",
                )
                exog_future = feature_store.extrapolate_features(combined_exog, future_index)
                if encoder:
                    exog_future = encoder.transform(exog_future)
                if scaler:
                    exog_future = scaler.transform(exog_future)

            if model.supports_multivariate and exog_future is not None:
                future_forecast = model.predict(horizon, exog_future)
            else:
                future_forecast = model.predict(horizon)

            # Save parameters
            params = model.get_params()
            param_store.save_params(model_name, category_name, params)

            # Compute feature importance if configured
            if compute_fi and combined_exog is not None:
                model_config_key = model_name.lower().replace(" ", "_")
                fi_config = models_config.get(model_config_key, {}).get(
                    "feature_importance", {}
                )

                if fi_config.get("enabled", False):
                    inner_model = getattr(model, "model", None)
                    if inner_model is not None:
                        fi_result = compute_feature_importance(
                            inner_model,
                            model_name,
                            combined_exog,
                            combined_target.iloc[: len(combined_exog)],
                            fi_config,
                        )
                        if fi_result:
                            feature_importance_result = fi_result

            # Get optuna trials used
            optuna_trials = params.get("optuna_trials_used", 0)

            # Build model result for JSON
            fi_for_json = None
            if feature_importance_result:
                if "shap" in feature_importance_result:
                    fi_for_json = feature_importance_result["shap"]
                elif "permutation" in feature_importance_result:
                    fi_for_json = feature_importance_result["permutation"]

            # Fallback to model's built-in feature importance
            if fi_for_json is None and hasattr(model, "get_feature_importance"):
                fi_for_json = model.get_feature_importance()

            model_json = build_model_result(
                metrics=metrics,
                params=params,
                retrained=do_optimize,
                optuna_trials_used=optuna_trials,
                feature_importance=fi_for_json,
            )

            return {
                "metrics": metrics,
                "test_pred": test_pred,
                "forecast": future_forecast,
                "model_json": model_json,
                "feature_importance": feature_importance_result,
            }

    except Exception as e:
        logger.error(f"Error training {model_name} for {category_name}: {e}")
        return None


def create_ensemble_with_fi(
    category_forecasts: dict[str, pd.Series],
    category_results: dict[str, dict],
    category_feature_importances: dict[str, dict],
    mape_threshold: float,
    max_models: int | None,
) -> tuple[pd.Series, dict[str, float], list[str], dict[str, float]]:
    """Create ensemble forecast with aggregated feature importance.

    Args:
        category_forecasts: Dictionary of model forecasts.
        category_results: Dictionary of model metrics.
        category_feature_importances: Dictionary of model feature importances.
        mape_threshold: MAPE threshold for ensemble inclusion.
        max_models: Maximum number of models in ensemble.

    Returns:
        Tuple of (ensemble_forecast, weights, models_used, ensemble_fi).
    """
    from forecast.evaluation.feature_importance import compute_ensemble_importance

    ensemble, weights, models_used = create_ensemble(
        category_forecasts, category_results, mape_threshold, max_models
    )

    # Compute ensemble feature importance
    ensemble_fi = {}
    if category_feature_importances and weights:
        # Extract SHAP or permutation importance
        model_imps = {}
        for model_name, imp_data in category_feature_importances.items():
            if "shap" in imp_data:
                model_imps[model_name] = imp_data["shap"]
            elif "permutation" in imp_data:
                model_imps[model_name] = imp_data["permutation"]

        if model_imps:
            ensemble_fi = compute_ensemble_importance(model_imps, weights)

    return ensemble, weights, models_used, ensemble_fi


def generate_plots(
    windowed_data: dict[str, pd.Series],
    cleaned_data: dict[str, pd.Series],
    change_logs: dict[str, list],
    splits_dict: dict[str, dict],
    all_test_actuals: dict[str, pd.Series],
    all_test_predictions: dict[str, dict[str, pd.Series]],
    all_results: dict[str, dict[str, dict]],
    all_forecasts: dict[str, dict[str, pd.Series]],
    ensemble_results: dict[str, dict],
    all_model_weights: dict[str, dict[str, float]],
    plots_dir: Path,
    plots_config: dict,
) -> None:
    """Generate all plots for the pipeline run.

    Args:
        windowed_data: Original windowed data.
        cleaned_data: Cleaned data.
        change_logs: Preprocessing change logs.
        splits_dict: Data splits.
        all_test_actuals: Test actual values.
        all_test_predictions: Test predictions by model.
        all_results: Model results/metrics.
        all_forecasts: Model forecasts.
        ensemble_results: Ensemble results.
        all_model_weights: Model weights for ensemble.
        plots_dir: Output directory for plots.
        plots_config: Plots configuration.
    """
    if plots_config.get("save_preprocessing", True):
        plot_all_preprocessing(
            original_dict=windowed_data,
            cleaned_dict=cleaned_data,
            change_logs=change_logs,
            splits_dict=splits_dict,
            output_dir=str(plots_dir / "preprocessing"),
            dpi=plots_config.get("figure_dpi", 150),
        )

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
                histories[name] = (
                    full_data["demand"]
                    if "demand" in full_data.columns
                    else full_data.iloc[:, 0]
                )
            else:
                histories[name] = full_data

        ensemble_forecasts = {
            name: info["forecast"] for name, info in ensemble_results.items()
        }

        plot_all_forecasts(
            histories=histories,
            forecasts=all_forecasts,
            ensemble_forecasts=ensemble_forecasts,
            ensemble_weights=all_model_weights,
            output_dir=str(plots_dir / "forecast"),
            dpi=plots_config.get("figure_dpi", 150),
        )


def write_outputs(
    all_results: dict[str, dict[str, dict]],
    all_test_actuals: dict[str, pd.Series],
    all_test_predictions: dict[str, dict[str, pd.Series]],
    ensemble_results: dict[str, dict],
    products_json_data: dict[str, dict],
    all_feature_importances: dict[str, dict[str, dict]],
    all_model_weights: dict[str, dict[str, float]],
    project_name: str,
    timestamp: str,
    results_dir: Path,
    config: dict,
    write_fi: bool = False,
) -> None:
    """Write all output files.

    Args:
        all_results: Model results/metrics.
        all_test_actuals: Test actual values.
        all_test_predictions: Test predictions.
        ensemble_results: Ensemble results.
        products_json_data: Product results for JSON.
        all_feature_importances: Feature importances by product/model.
        all_model_weights: Model weights.
        project_name: Project name.
        timestamp: Run timestamp.
        results_dir: Output directory.
        config: Full configuration.
        write_fi: Whether to write feature importance JSON.
    """
    logger = get_logger()

    write_model_results(
        all_results,
        all_test_actuals,
        all_test_predictions,
        str(results_dir / "model_results.xlsx"),
    )

    write_ensemble_results(ensemble_results, str(results_dir / "ensemble_forecasts.xlsx"))

    # Write feature importance if computed and requested
    if write_fi and all_feature_importances:
        from forecast.evaluation.feature_importance import write_ensemble_fi_with_weights

        write_ensemble_fi_with_weights(
            project_name=project_name,
            all_importances=all_feature_importances,
            model_weights=all_model_weights,
            output_dir=results_dir,
        )

    # Write consolidated JSON summary
    write_run_summary(
        project_name=project_name,
        timestamp=timestamp,
        results_dir=results_dir,
        products_data=products_json_data,
        run_config={
            "models": config.get("models", {}),
            "pipeline": config.get("pipeline", {}),
            "preprocessing": config.get("preprocessing", {}),
        },
    )

    logger.info(f"Results saved to: {results_dir}")
