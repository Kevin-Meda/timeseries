"""Multivariate forecasting pipeline with cross-product features."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from forecast.models import create_multivariate_models
from forecast.models.base import BaseForecaster
from forecast.evaluation import evaluate_all, compute_model_feature_importance
from forecast.output import build_product_result, build_model_result
from forecast.tuning.param_store import ParamStore, should_optimize
from forecast.utils import get_logger

from forecast.pipeline_common import (
    load_config,
    validate_multivariate_config,
    setup_pipeline_run,
    load_and_preprocess_data,
    load_feature_store,
    build_cross_product_features,
    split_data,
    extract_target,
    get_exog_features,
    encode_and_scale_features,
    create_ensemble_with_fi,
    generate_plots,
    write_outputs,
)


def _extrapolate_encoded_features(
    combined_exog: pd.DataFrame,
    future_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Extrapolate already-encoded features for future periods.

    Uses last known values from the encoded feature DataFrame.

    Args:
        combined_exog: DataFrame with encoded features.
        future_index: DatetimeIndex for future periods.

    Returns:
        DataFrame with extrapolated feature values.
    """
    if combined_exog.empty or len(future_index) == 0:
        return pd.DataFrame(index=future_index)

    # Get last known values for each feature
    last_values = combined_exog.iloc[-1]

    # Create future DataFrame with last values repeated
    future_df = pd.DataFrame(
        {col: [last_values[col]] * len(future_index) for col in combined_exog.columns},
        index=future_index,
    )

    return future_df


def _create_fresh_multivariate_model(
    template: BaseForecaster, config: dict
) -> BaseForecaster:
    """Create a fresh multivariate model instance based on a template.

    Args:
        template: Model instance to copy settings from.
        config: Models configuration dictionary.

    Returns:
        New model instance.
    """
    from forecast.models import ProphetForecaster, XGBoostForecaster, ChronosForecaster
    from forecast.models.xgboost import is_xgboost_available
    from forecast.models.prophet import is_prophet_available

    if template.name == "Prophet" and is_prophet_available():
        prophet_config = config.get("prophet", {})
        defaults = prophet_config.get("defaults", {})
        optimize = prophet_config.get("optimize_params", True)
        return ProphetForecaster(
            use_optuna=optimize,
            optuna_trials=prophet_config.get("optuna_trials", 5),
            changepoint_prior_scale=defaults.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=defaults.get("seasonality_prior_scale", 10.0),
            seasonality_mode=defaults.get("seasonality_mode", "additive"),
        )

    elif template.name == "XGBoost" and is_xgboost_available():
        xgb_config = config.get("xgboost", {})
        defaults = xgb_config.get("defaults", {})
        optimize = xgb_config.get("optimize_params", True)
        return XGBoostForecaster(
            use_optuna=optimize,
            optuna_trials=xgb_config.get("optuna_trials", 5),
            n_estimators=defaults.get("n_estimators", 100),
            max_depth=defaults.get("max_depth", 6),
            learning_rate=defaults.get("learning_rate", 0.1),
        )

    elif template.name == "Chronos":
        chronos_config = config.get("chronos", {})
        defaults = chronos_config.get("defaults", {})
        model_name = defaults.get("model_name", "amazon/chronos-2")
        return ChronosForecaster(model_name=model_name)

    return template


class CrossProductFeatureStore:
    """Feature store wrapper that extrapolates cross-product features."""

    def __init__(
        self,
        base_feature_store,
        cleaned_data: dict[str, pd.Series],
    ):
        """Initialize cross-product feature store.

        Args:
            base_feature_store: Original feature store loader.
            cleaned_data: Dictionary of cleaned demand series by product.
        """
        self.base_store = base_feature_store
        self.cleaned_data = cleaned_data
        self.products = list(cleaned_data.keys())

    def extrapolate_features(
        self,
        combined_exog: pd.DataFrame,
        future_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Extrapolate features including cross-product demand.

        Args:
            combined_exog: Combined historical exogenous features.
            future_index: Future date index.

        Returns:
            Extrapolated features DataFrame.
        """
        # Start with base feature store extrapolation if available
        if self.base_store is not None:
            # Filter to non-demand columns for base extrapolation
            base_cols = [c for c in combined_exog.columns if not c.endswith("_demand")]
            if base_cols:
                base_exog = combined_exog[base_cols]
                result = self.base_store.extrapolate_features(base_exog, future_index)
            else:
                result = pd.DataFrame(index=future_index)
        else:
            result = pd.DataFrame(index=future_index)

        # Extrapolate cross-product demand features using simple forward fill / mean
        demand_cols = [c for c in combined_exog.columns if c.endswith("_demand")]
        for col in demand_cols:
            # Use recent mean as forecast for other products' demand
            recent_values = combined_exog[col].iloc[-12:]  # Last 12 months
            result[col] = recent_values.mean()

        return result


def run_multivariate_all_pipeline(
    config_dir: str = "configs",
    input_override: str | None = None,
    output_dir_override: str | None = None,
    log_level: str = "INFO",
    project_name: str = "default",
) -> dict:
    """Run the multivariate all-products features pipeline.

    Uses Prophet, XGBoost, and Chronos with features from feature store
    PLUS other products' demand as features.
    Outputs to: output/multivariate_all/{project}/

    Args:
        config_dir: Path to configuration directory.
        input_override: Optional override for input file path.
        output_dir_override: Optional override for output directory.
        log_level: Logging level.
        project_name: Project name for organizing outputs.

    Returns:
        Dictionary with pipeline results.
    """
    # Setup pipeline run
    logger, pm, timestamp = setup_pipeline_run(
        project_name=project_name,
        log_level=log_level,
        output_dir_override=output_dir_override,
        pipeline_type="multivariate_all",
    )

    # Load and validate configuration
    config = load_config(config_dir)
    logger.info(f"Loaded configuration from: {config_dir}")

    try:
        validate_multivariate_config(config)
    except ValueError as e:
        logger.error(str(e))
        return {"error": str(e)}

    data_config = config.get("data_input", {})
    preprocess_config = config.get("preprocessing", {})
    models_config = config.get("models", {})
    pipeline_config = config.get("pipeline", {})
    plots_config = config.get("plots", {})

    # Get directories
    results_dir = pm.results_dir
    plots_dir = pm.plots_dir

    # Pipeline settings
    categorical_features = data_config.get("categorical_features", [])
    horizon = pipeline_config.get("forecast_horizon", 12)
    mape_threshold = pipeline_config.get("ensemble_mape_threshold", 0.5)
    max_models = pipeline_config.get("ensemble_max_models")

    # Scaler settings
    scaler_config = preprocess_config.get("scaler", {})
    use_scaler = scaler_config.get("enabled", False)
    scaler_type = scaler_config.get("type", "robust")

    # Load and preprocess data
    try:
        cleaned_data, windowed_data, change_logs = load_and_preprocess_data(
            config, input_override
        )
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {e}")
        return {"error": str(e)}
    except ValueError as e:
        logger.error(str(e))
        return {"error": str(e)}

    # Load feature store
    feature_store, base_features_by_product = load_feature_store(config, cleaned_data)

    # Build cross-product features
    logger.info("Building cross-product features (other products as features)")
    features_by_product = build_cross_product_features(
        cleaned_data, base_features_by_product
    )

    if not features_by_product:
        logger.error("Failed to build cross-product features")
        return {"error": "Failed to build cross-product features"}

    for product, features in features_by_product.items():
        logger.info(f"  {product}: {len(features.columns) - 1} features")

    # Create cross-product feature store for extrapolation
    cross_product_store = CrossProductFeatureStore(feature_store, cleaned_data)

    # Split data with features
    splits_dict = split_data(cleaned_data, features_by_product, config)

    # Create multivariate models
    models_list = create_multivariate_models(models_config)
    if not models_list:
        logger.error(
            "No multivariate models enabled. "
            "Enable prophet, xgboost, or chronos in models.yaml"
        )
        return {"error": "No multivariate models enabled"}

    logger.info(f"Training multivariate models: {[m.name for m in models_list]}")

    # Initialize result containers
    all_results: dict[str, dict[str, dict]] = {}
    all_test_predictions: dict[str, dict[str, pd.Series]] = {}
    all_test_actuals: dict[str, pd.Series] = {}
    all_forecasts: dict[str, dict[str, pd.Series]] = {}
    ensemble_results: dict[str, dict] = {}
    products_json_data: dict[str, dict] = {}
    all_model_weights: dict[str, dict[str, float]] = {}
    all_feature_importances: dict[str, dict[str, dict]] = {}

    # Initialize param store
    param_store = ParamStore(pm)

    # Process each product
    for category_name, splits in splits_dict.items():
        logger.info(f"Processing category: {category_name}")

        train = splits["train"]
        val = splits["val"]
        test = splits["test"]

        # Extract targets
        train_target = extract_target(train)
        val_target = extract_target(val)
        test_target = extract_target(test)

        # Extract exogenous features
        exog_train = get_exog_features(train) if isinstance(train, pd.DataFrame) else None
        exog_val = get_exog_features(val) if isinstance(val, pd.DataFrame) else None
        exog_test = get_exog_features(test) if isinstance(test, pd.DataFrame) else None

        # Apply encoding and scaling
        exog_train, exog_val, exog_test, encoder, scaler = encode_and_scale_features(
            exog_train, exog_val, exog_test, categorical_features, use_scaler, scaler_type
        )

        category_results: dict[str, dict] = {}
        category_test_predictions: dict[str, pd.Series] = {}
        category_forecasts: dict[str, pd.Series] = {}
        category_model_json: dict[str, dict] = {}
        category_feature_importances: dict[str, dict] = {}

        # Train each model
        for model in models_list:
            model_instance = _create_fresh_multivariate_model(model, models_config)
            model_name = model_instance.name

            logger.debug(f"Training {model_name} for {category_name}")

            try:
                # Check if we should optimize or load params
                do_optimize, loaded_params = should_optimize(
                    models_config, pm, model_name, category_name
                )

                if not do_optimize and loaded_params:
                    model_instance.load_params(loaded_params)
                    logger.info(f"Loaded params for {model_name}/{category_name}")

                # Fit model with exogenous features
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
                    if model_instance.supports_multivariate and combined_exog is not None:
                        last_date = combined_target.index[-1]
                        future_index = pd.date_range(
                            start=last_date + pd.DateOffset(months=1),
                            periods=horizon,
                            freq="MS",
                        )
                        # Extrapolate from already-encoded combined_exog
                        # No need to re-encode/re-scale since values are already transformed
                        exog_future = _extrapolate_encoded_features(
                            combined_exog, future_index
                        )

                    if model_instance.supports_multivariate and exog_future is not None:
                        future_forecast = model_instance.predict(horizon, exog_future)
                    else:
                        future_forecast = model_instance.predict(horizon)

                    category_forecasts[model_name] = future_forecast

                    # Save parameters
                    params = model_instance.get_params()
                    param_store.save_params(model_name, category_name, params)

                    # Compute feature importance using model-specific methods
                    feature_importance_result = None
                    model_config_key = model_name.lower().replace(" ", "_")
                    fi_config = models_config.get(model_config_key, {}).get(
                        "feature_importance", {}
                    )

                    if fi_config.get("enabled", False) and combined_exog is not None:
                        fi_result = compute_model_feature_importance(
                            model_instance,
                            model_name,
                            combined_exog,
                            combined_target.iloc[: len(combined_exog)],
                            fi_config,
                        )
                        if fi_result:
                            feature_importance_result = fi_result
                            category_feature_importances[model_name] = fi_result
                            logger.info(f"Computed feature importance for {model_name}")

                    optuna_trials = params.get("optuna_trials_used", 0)

                    # Build feature importance for JSON (priority: builtin > shap > permutation)
                    fi_for_json = None
                    if feature_importance_result:
                        for fi_type in ["builtin", "shap", "permutation", "regressor_coefficients"]:
                            if fi_type in feature_importance_result:
                                fi_for_json = feature_importance_result[fi_type]
                                break

                    category_model_json[model_name] = build_model_result(
                        metrics=metrics,
                        params=params,
                        retrained=do_optimize,
                        optuna_trials_used=optuna_trials,
                        feature_importance=fi_for_json,
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

            # Store feature importances
            if category_feature_importances:
                all_feature_importances[category_name] = category_feature_importances

            # Create ensemble with feature importance
            if len(category_forecasts) > 0:
                ensemble, weights, models_used, ensemble_fi = create_ensemble_with_fi(
                    category_forecasts,
                    category_results,
                    category_feature_importances,
                    mape_threshold,
                    max_models,
                )

                # Build complete weights dict
                complete_weights = {}
                for model_name in category_forecasts.keys():
                    complete_weights[model_name] = weights.get(model_name, 0.0)

                all_model_weights[category_name] = complete_weights

                if len(ensemble) > 0:
                    ensemble_results[category_name] = {
                        "forecast": ensemble,
                        "mape": category_results.get(models_used[0], {}).get("mape", 0)
                        if models_used
                        else 0,
                        "models_used": models_used,
                        "weights": weights,
                        "ensemble_feature_importance": ensemble_fi,
                    }

    # Generate plots
    generate_plots(
        windowed_data=windowed_data,
        cleaned_data=cleaned_data,
        change_logs=change_logs,
        splits_dict=splits_dict,
        all_test_actuals=all_test_actuals,
        all_test_predictions=all_test_predictions,
        all_results=all_results,
        all_forecasts=all_forecasts,
        ensemble_results=ensemble_results,
        all_model_weights=all_model_weights,
        plots_dir=plots_dir,
        plots_config=plots_config,
    )

    # Write outputs with feature importance
    write_outputs(
        all_results=all_results,
        all_test_actuals=all_test_actuals,
        all_test_predictions=all_test_predictions,
        ensemble_results=ensemble_results,
        products_json_data=products_json_data,
        all_feature_importances=all_feature_importances,
        all_model_weights=all_model_weights,
        project_name=project_name,
        timestamp=timestamp,
        results_dir=results_dir,
        config=config,
        write_fi=True,
    )

    logger.info("Multivariate all-products pipeline completed successfully")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Plots saved to: {plots_dir}")

    return {
        "model_results": all_results,
        "ensemble_results": ensemble_results,
        "forecasts": all_forecasts,
        "feature_importances": all_feature_importances,
        "categories_processed": len(all_results),
        "timestamp": timestamp,
        "project_name": project_name,
        "pipeline_type": "multivariate_all",
    }


def main() -> int:
    """CLI entry point for multivariate all-products pipeline."""
    parser = argparse.ArgumentParser(
        description="Multivariate All-Products Pipeline - Prophet, XGBoost, Chronos with cross-product features",
    )

    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Path to configuration directory (default: configs)",
    )
    parser.add_argument(
        "--input",
        dest="input_file",
        help="Override input Excel file path",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Override output directory for results",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--project-name",
        dest="project_name",
        default="default",
        help="Project name for organizing outputs (default: default)",
    )

    args = parser.parse_args()

    try:
        results = run_multivariate_all_pipeline(
            config_dir=args.config_dir,
            input_override=args.input_file,
            output_dir_override=args.output_dir,
            log_level=args.log_level,
            project_name=args.project_name,
        )

        if "error" in results:
            print(f"Pipeline failed: {results['error']}", file=sys.stderr)
            return 1

        print(f"\nMultivariate all-products pipeline completed successfully!")
        print(f"Categories processed: {results.get('categories_processed', 0)}")
        return 0

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Pipeline failed with error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
