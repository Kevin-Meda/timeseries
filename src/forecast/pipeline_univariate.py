"""Univariate forecasting pipeline using SARIMA and Holt-Winters."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from forecast.models import create_univariate_models
from forecast.models.base import BaseForecaster
from forecast.evaluation import create_ensemble
from forecast.output import build_product_result, build_model_result
from forecast.tuning.param_store import ParamStore, should_optimize
from forecast.utils import get_logger

from forecast.pipeline_common import (
    load_config,
    validate_univariate_config,
    setup_pipeline_run,
    load_and_preprocess_data,
    split_data,
    extract_target,
    generate_plots,
    write_outputs,
)


def _create_fresh_univariate_model(
    template: BaseForecaster, config: dict
) -> BaseForecaster:
    """Create a fresh univariate model instance based on a template.

    Args:
        template: Model instance to copy settings from.
        config: Models configuration dictionary.

    Returns:
        New model instance.
    """
    from forecast.models import SARIMAForecaster, HoltWintersForecaster

    if template.name == "SARIMA":
        sarima_config = config.get("sarima", {})
        optimize = sarima_config.get("optimize_params", True)
        return SARIMAForecaster(
            use_optuna=optimize,
            optuna_trials=sarima_config.get("optuna_trials", 5),
        )

    elif template.name == "HoltWinters":
        hw_config = config.get("holt_winters", {})
        defaults = hw_config.get("defaults", {})
        return HoltWintersForecaster(
            trend=defaults.get("trend", "add"),
            seasonal=defaults.get("seasonal", "add"),
            seasonal_periods=defaults.get("seasonal_periods", 12),
        )

    return template


def run_univariate_pipeline(
    config_dir: str = "configs",
    input_override: str | None = None,
    output_dir_override: str | None = None,
    log_level: str = "INFO",
    project_name: str = "default",
) -> dict:
    """Run the univariate forecasting pipeline.

    Uses only SARIMA and Holt-Winters models without exogenous features.
    Outputs to: output/univariate/{project}/

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
        pipeline_type="univariate",
    )

    # Load and validate configuration
    config = load_config(config_dir)
    logger.info(f"Loaded configuration from: {config_dir}")

    try:
        validate_univariate_config(config)
    except ValueError as e:
        logger.error(str(e))
        return {"error": str(e)}

    models_config = config.get("models", {})
    pipeline_config = config.get("pipeline", {})
    plots_config = config.get("plots", {})

    # Get directories
    results_dir = pm.results_dir
    plots_dir = pm.plots_dir

    # Pipeline settings
    horizon = pipeline_config.get("forecast_horizon", 12)
    mape_threshold = pipeline_config.get("ensemble_mape_threshold", 0.5)
    max_models = pipeline_config.get("ensemble_max_models")

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

    # Split data (no features for univariate)
    splits_dict = split_data(cleaned_data, {}, config)

    # Create univariate models
    models_list = create_univariate_models(models_config)
    if not models_list:
        logger.error("No univariate models enabled. Enable sarima or holt_winters in models.yaml")
        return {"error": "No univariate models enabled"}

    logger.info(f"Training univariate models: {[m.name for m in models_list]}")

    # Initialize result containers
    all_results: dict[str, dict[str, dict]] = {}
    all_test_predictions: dict[str, dict[str, pd.Series]] = {}
    all_test_actuals: dict[str, pd.Series] = {}
    all_forecasts: dict[str, dict[str, pd.Series]] = {}
    ensemble_results: dict[str, dict] = {}
    products_json_data: dict[str, dict] = {}
    all_model_weights: dict[str, dict[str, float]] = {}

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

        category_results: dict[str, dict] = {}
        category_test_predictions: dict[str, pd.Series] = {}
        category_forecasts: dict[str, pd.Series] = {}
        category_model_json: dict[str, dict] = {}

        # Train each model
        for model in models_list:
            model_instance = _create_fresh_univariate_model(model, models_config)
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

                # Fit model (univariate - no exog)
                model_instance.fit(train_target, val_target)

                if not model_instance.is_fitted:
                    logger.warning(f"{model_name} failed to fit for {category_name}")
                    continue

                # Predict on test set
                test_pred = model_instance.predict(len(test_target))

                # Align indices
                if hasattr(test_pred, "index") and hasattr(test_target, "index"):
                    test_pred.index = test_target.index

                # Evaluate
                from forecast.evaluation import evaluate_all

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
                model_instance.fit(combined_target, None)

                if model_instance.is_fitted:
                    future_forecast = model_instance.predict(horizon)
                    category_forecasts[model_name] = future_forecast

                    # Save parameters
                    params = model_instance.get_params()
                    param_store.save_params(model_name, category_name, params)

                    optuna_trials = params.get("optuna_trials_used", 0)

                    category_model_json[model_name] = build_model_result(
                        metrics=metrics,
                        params=params,
                        retrained=do_optimize,
                        optuna_trials_used=optuna_trials,
                        feature_importance=None,  # No FI for univariate
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

            # Create ensemble
            if len(category_forecasts) > 0:
                ensemble, weights, models_used = create_ensemble(
                    category_forecasts, category_results, mape_threshold, max_models
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

    # Write outputs (no feature importance for univariate)
    write_outputs(
        all_results=all_results,
        all_test_actuals=all_test_actuals,
        all_test_predictions=all_test_predictions,
        ensemble_results=ensemble_results,
        products_json_data=products_json_data,
        all_feature_importances={},
        all_model_weights=all_model_weights,
        project_name=project_name,
        timestamp=timestamp,
        results_dir=results_dir,
        config=config,
        write_fi=False,
    )

    logger.info("Univariate pipeline completed successfully")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"Plots saved to: {plots_dir}")

    return {
        "model_results": all_results,
        "ensemble_results": ensemble_results,
        "forecasts": all_forecasts,
        "categories_processed": len(all_results),
        "timestamp": timestamp,
        "project_name": project_name,
        "pipeline_type": "univariate",
    }


def main() -> int:
    """CLI entry point for univariate pipeline."""
    parser = argparse.ArgumentParser(
        description="Univariate Forecasting Pipeline - SARIMA and Holt-Winters",
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
        results = run_univariate_pipeline(
            config_dir=args.config_dir,
            input_override=args.input_file,
            output_dir_override=args.output_dir,
            log_level=args.log_level,
            project_name=args.project_name,
        )

        if "error" in results:
            print(f"Pipeline failed: {results['error']}", file=sys.stderr)
            return 1

        print(f"\nUnivariate pipeline completed successfully!")
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
