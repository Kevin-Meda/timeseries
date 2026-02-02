"""Feature importance computation for forecasting models."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.utils import get_logger


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, float]:
    """Compute permutation importance for a fitted model.

    Measures feature importance by shuffling each feature and measuring
    the decrease in model performance.

    Args:
        model: Fitted model with a predict method.
        X: Feature DataFrame used for evaluation.
        y: Target values.
        n_repeats: Number of times to permute each feature.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping feature names to importance scores (percentages).
    """
    logger = get_logger()

    if not hasattr(model, "predict"):
        logger.warning("Model does not have predict method, skipping permutation importance")
        return {}

    rng = np.random.RandomState(random_state)

    # Get baseline prediction error
    try:
        baseline_pred = model.predict(X)
        baseline_error = np.mean(np.abs((y.values - baseline_pred) / (y.values + 1e-8)))
    except Exception as e:
        logger.warning(f"Failed to compute baseline prediction: {e}")
        return {}

    importances = {}

    for col in X.columns:
        errors = []
        X_permuted = X.copy()

        for _ in range(n_repeats):
            # Shuffle the column
            X_permuted[col] = rng.permutation(X[col].values)

            try:
                pred = model.predict(X_permuted)
                error = np.mean(np.abs((y.values - pred) / (y.values + 1e-8)))
                errors.append(error)
            except Exception:
                continue

            # Restore original values
            X_permuted[col] = X[col].values

        if errors:
            # Importance is the mean increase in error when feature is permuted
            mean_error = np.mean(errors)
            importances[col] = max(0, mean_error - baseline_error)

    # Normalize to sum to 100 (percentages)
    total = sum(importances.values())
    if total > 0:
        importances = {k: round(float(v / total) * 100, 2) for k, v in importances.items()}

    return importances


def compute_shap_importance(
    model: Any,
    X: pd.DataFrame,
    model_type: str = "tree",
    max_samples: int = 100,
) -> dict[str, float]:
    """Compute SHAP-based feature importance.

    Uses TreeSHAP for tree-based models, KernelSHAP for others.

    Args:
        model: Fitted model.
        X: Feature DataFrame used for computing SHAP values.
        model_type: Type of model - "tree" for TreeSHAP, "other" for KernelSHAP.
        max_samples: Maximum samples to use for KernelSHAP (for performance).

    Returns:
        Dictionary mapping feature names to SHAP importance scores (percentages).
    """
    logger = get_logger()

    try:
        import shap
    except ImportError:
        logger.warning("SHAP not available. Install with: pip install shap")
        return {}

    try:
        if model_type == "tree":
            # Use TreeSHAP (fast for tree-based models)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        else:
            # Use KernelSHAP (works for any model but slower)
            # Subsample for performance
            if len(X) > max_samples:
                X_sample = X.sample(n=max_samples, random_state=42)
            else:
                X_sample = X

            # Create background dataset
            background = shap.sample(X_sample, min(10, len(X_sample)))
            explainer = shap.KernelExplainer(model.predict, background)
            shap_values = explainer.shap_values(X_sample)

        # Compute mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create importance dict (convert numpy to native Python float)
        importances = {col: float(val) for col, val in zip(X.columns, mean_abs_shap)}

        # Normalize to sum to 100 (percentages)
        total = sum(importances.values())
        if total > 0:
            importances = {k: round(float(v / total) * 100, 2) for k, v in importances.items()}

        return importances

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        return {}


def compute_xgboost_feature_importance(
    model_instance: Any,
    fi_config: dict,
) -> dict[str, dict[str, float]]:
    """Compute feature importance for XGBoost model.

    Uses built-in feature_importances_ and optionally SHAP.

    Args:
        model_instance: XGBoost forecaster instance.
        fi_config: Feature importance configuration.

    Returns:
        Dictionary with "builtin" and optionally "shap" importance dicts.
    """
    logger = get_logger()
    result = {}

    if not hasattr(model_instance, "model") or model_instance.model is None:
        return result

    model = model_instance.model
    feature_columns = getattr(model_instance, "_feature_columns", [])

    if not feature_columns:
        return result

    # Built-in feature importance (gain-based)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        total = float(sum(importances))
        if total > 0:
            result["builtin"] = {
                col: round(float(imp / total) * 100, 2)
                for col, imp in zip(feature_columns, importances)
            }
            logger.debug(f"Computed XGBoost built-in feature importance")

    # SHAP importance
    if fi_config.get("shap", False):
        try:
            import shap

            # Get training data features
            train_data = getattr(model_instance, "_train_data", None)
            if train_data is not None:
                # Prepare features same way as training
                all_features = model_instance._prepare_features(train_data, is_training=True)
                X = all_features[feature_columns]

                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                mean_abs_shap = np.abs(shap_values).mean(axis=0)

                total = float(sum(mean_abs_shap))
                if total > 0:
                    result["shap"] = {
                        col: round(float(imp / total) * 100, 2)
                        for col, imp in zip(feature_columns, mean_abs_shap)
                    }
                    logger.debug(f"Computed XGBoost SHAP feature importance")

        except ImportError:
            logger.debug("SHAP not available for XGBoost")
        except Exception as e:
            logger.warning(f"SHAP computation failed for XGBoost: {e}")

    return result


def compute_prophet_feature_importance(
    model_instance: Any,
    fi_config: dict,
) -> dict[str, dict[str, float]]:
    """Compute feature importance for Prophet model.

    Uses regressor coefficients from the fitted model.

    Args:
        model_instance: Prophet forecaster instance.
        fi_config: Feature importance configuration.

    Returns:
        Dictionary with "regressor_coefficients" importance dict.
    """
    logger = get_logger()
    result = {}

    if not hasattr(model_instance, "model") or model_instance.model is None:
        return result

    model = model_instance.model
    regressor_columns = getattr(model_instance, "_regressor_columns", [])

    if not regressor_columns:
        logger.debug("Prophet has no regressors, skipping FI")
        return result

    # Extract regressor coefficients from Prophet model
    try:
        # Prophet stores regressor info in model.extra_regressors
        if hasattr(model, "extra_regressors") and model.extra_regressors:
            coefs = {}
            for reg_name in regressor_columns:
                if reg_name in model.extra_regressors:
                    # Get the coefficient (beta) for this regressor
                    reg_data = model.extra_regressors[reg_name]
                    # The actual coefficient is stored in the params
                    if hasattr(model, "params") and "beta" in model.params:
                        # Find index of this regressor
                        reg_idx = list(model.extra_regressors.keys()).index(reg_name)
                        if reg_idx < len(model.params["beta"]):
                            coef = abs(float(model.params["beta"][reg_idx].mean()))
                            coefs[reg_name] = coef

            if coefs:
                total = float(sum(coefs.values()))
                if total > 0:
                    result["regressor_coefficients"] = {
                        k: round(float(v / total) * 100, 2) for k, v in coefs.items()
                    }
                    logger.debug(f"Computed Prophet regressor coefficients")

    except Exception as e:
        logger.warning(f"Failed to extract Prophet regressor coefficients: {e}")

    # Permutation importance as alternative
    if fi_config.get("permutation", False) and not result:
        try:
            train_data = getattr(model_instance, "_train_data", None)
            if train_data is not None and regressor_columns:
                # Create Prophet-format DataFrame
                prophet_df = pd.DataFrame()
                prophet_df["ds"] = train_data.index
                prophet_df["y"] = train_data["demand"].values
                for col in regressor_columns:
                    if col in train_data.columns:
                        prophet_df[col] = train_data[col].values

                X = prophet_df[regressor_columns].copy()
                y = prophet_df["y"]

                # Create a wrapper that predicts using Prophet
                class ProphetWrapper:
                    def __init__(self, prophet_model, ds_values, regressor_cols):
                        self.model = prophet_model
                        self.ds_values = ds_values
                        self.regressor_cols = regressor_cols

                    def predict(self, X_df):
                        future = pd.DataFrame({"ds": self.ds_values})
                        for col in self.regressor_cols:
                            future[col] = X_df[col].values
                        forecast = self.model.predict(future)
                        return forecast["yhat"].values

                wrapper = ProphetWrapper(model, prophet_df["ds"].values, regressor_columns)
                perm_importance = compute_permutation_importance(wrapper, X, y, n_repeats=5)
                if perm_importance:
                    result["permutation"] = perm_importance
                    logger.debug(f"Computed Prophet permutation importance")

        except Exception as e:
            logger.warning(f"Failed to compute Prophet permutation importance: {e}")

    return result


def compute_chronos_feature_importance(
    model_instance: Any,
    combined_exog: pd.DataFrame,
    combined_target: pd.Series,
    fi_config: dict,
) -> dict[str, dict[str, float]]:
    """Compute feature importance for Chronos model.

    Since Chronos may not support covariates in all versions,
    we use permutation importance on the prediction output.

    Args:
        model_instance: Chronos forecaster instance.
        combined_exog: Combined exogenous features.
        combined_target: Combined target series.
        fi_config: Feature importance configuration.

    Returns:
        Dictionary with "permutation" importance dict.
    """
    logger = get_logger()
    result = {}

    # Check if Chronos is using covariates
    covariates = getattr(model_instance, "_covariates", None)
    if covariates is None or len(covariates.columns) == 0:
        logger.debug("Chronos not using covariates, skipping FI")
        return result

    if not fi_config.get("permutation", False):
        return result

    # For Chronos, FI is limited since it's a pretrained model
    # We can note which features were provided but can't compute true importance
    # unless the version supports covariates
    logger.debug("Chronos feature importance limited - pretrained model")

    # Return the feature names with equal weights as placeholder
    feature_cols = list(covariates.columns)
    if feature_cols:
        equal_weight = round(float(100.0 / len(feature_cols)), 2)
        result["covariates_used"] = {col: float(equal_weight) for col in feature_cols}

    return result


def compute_model_feature_importance(
    model_instance: Any,
    model_name: str,
    combined_exog: pd.DataFrame | None,
    combined_target: pd.Series,
    fi_config: dict,
) -> dict[str, dict[str, float]]:
    """Compute feature importance for any supported model.

    Routes to model-specific FI computation.

    Args:
        model_instance: Model forecaster instance.
        model_name: Name of the model.
        combined_exog: Combined exogenous features.
        combined_target: Combined target series.
        fi_config: Feature importance configuration.

    Returns:
        Dictionary with importance type -> feature -> percentage mapping.
    """
    logger = get_logger()

    if not fi_config.get("enabled", False):
        return {}

    if combined_exog is None or len(combined_exog.columns) == 0:
        return {}

    model_name_lower = model_name.lower()

    if "xgboost" in model_name_lower:
        return compute_xgboost_feature_importance(model_instance, fi_config)
    elif "prophet" in model_name_lower:
        return compute_prophet_feature_importance(model_instance, fi_config)
    elif "chronos" in model_name_lower:
        return compute_chronos_feature_importance(
            model_instance, combined_exog, combined_target, fi_config
        )
    else:
        # Generic permutation importance for other models
        inner_model = getattr(model_instance, "model", None)
        if inner_model is not None and hasattr(inner_model, "predict"):
            perm_imp = compute_permutation_importance(
                inner_model, combined_exog, combined_target
            )
            if perm_imp:
                return {"permutation": perm_imp}

    return {}


def compute_feature_importance(
    model: Any,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute feature importance for a model based on config.

    DEPRECATED: Use compute_model_feature_importance instead.

    Args:
        model: Fitted model.
        model_name: Name of the model (e.g., "XGBoost", "Prophet").
        X: Feature DataFrame.
        y: Target values.
        config: Feature importance configuration with keys:
            - enabled: bool (default True)
            - permutation: bool (default True)
            - shap: bool (default True)

    Returns:
        Dictionary with "permutation" and/or "shap" importance dicts.
    """
    logger = get_logger()

    if not config.get("enabled", True):
        return {}

    result = {}

    # Determine model type for SHAP
    model_type = "tree" if model_name in ["XGBoost", "LightGBM", "RandomForest"] else "other"

    # Compute permutation importance
    if config.get("permutation", True):
        logger.debug(f"Computing permutation importance for {model_name}")
        perm_importance = compute_permutation_importance(model, X, y)
        if perm_importance:
            result["permutation"] = perm_importance

    # Compute SHAP importance
    if config.get("shap", True):
        logger.debug(f"Computing SHAP importance for {model_name}")
        shap_importance = compute_shap_importance(model, X, model_type)
        if shap_importance:
            result["shap"] = shap_importance

    return result


def compute_ensemble_importance(
    model_importances: dict[str, dict[str, float]],
    model_weights: dict[str, float],
) -> dict[str, float]:
    """Compute weighted ensemble feature importance.

    Averages feature importances across models using the same weights
    as the ensemble forecast (inverse MAPE weights).

    Args:
        model_importances: Dictionary mapping model names to importance dicts.
            Each importance dict maps feature names to importance scores.
        model_weights: Dictionary mapping model names to ensemble weights.

    Returns:
        Dictionary mapping feature names to weighted average importance (percentages).
    """
    logger = get_logger()

    if not model_importances or not model_weights:
        return {}

    # Collect all feature names across models
    all_features = set()
    for importance_dict in model_importances.values():
        all_features.update(importance_dict.keys())

    if not all_features:
        return {}

    # Compute weighted average for each feature
    ensemble_importance = {}
    total_weight = float(sum(
        float(weight) for model, weight in model_weights.items()
        if model in model_importances
    ))

    if total_weight == 0:
        logger.warning("No matching models between importances and weights")
        return {}

    for feature in all_features:
        weighted_sum = 0.0
        for model, weight in model_weights.items():
            if model in model_importances and feature in model_importances[model]:
                weighted_sum += float(weight) * float(model_importances[model][feature])
        ensemble_importance[feature] = float(weighted_sum / total_weight)

    # Normalize to sum to 100 (percentages)
    total = float(sum(ensemble_importance.values()))
    if total > 0:
        ensemble_importance = {k: round(float(v / total) * 100, 2) for k, v in ensemble_importance.items()}

    return ensemble_importance


def write_project_feature_importance(
    project_name: str,
    all_importances: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path | str,
    ensemble_weights: dict[str, dict[str, float]] | None = None,
) -> Path:
    """Write project-level feature importance to JSON.

    Output structure:
    {
        "project": "project_name",
        "products": {
            "Product_A": {
                "models": {
                    "XGBoost": {
                        "permutation": {"feature1": 0.15, ...},
                        "shap": {"feature1": 0.12, ...}
                    }
                }
            }
        },
        "ensemble": {
            "Product_A": {"feature1": 0.13, ...}
        }
    }

    Args:
        project_name: Name of the project.
        all_importances: Nested dict of product -> model -> importance_type -> importance.
        output_dir: Directory to save the JSON file.
        ensemble_weights: Optional dict of product -> model -> weight for ensemble importance.

    Returns:
        Path to the saved JSON file.
    """
    logger = get_logger()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "project": project_name,
        "products": {},
        "ensemble": {},
    }

    for product_name, model_data in all_importances.items():
        result["products"][product_name] = {"models": model_data}

        # Compute ensemble importance if weights provided
        if ensemble_weights and product_name in ensemble_weights:
            product_weights = ensemble_weights[product_name]

            # Extract SHAP or permutation importance for ensemble
            # Prefer SHAP, fallback to permutation
            model_imps = {}
            for model_name, imp_data in model_data.items():
                if "shap" in imp_data:
                    model_imps[model_name] = imp_data["shap"]
                elif "permutation" in imp_data:
                    model_imps[model_name] = imp_data["permutation"]

            if model_imps:
                ensemble_imp = compute_ensemble_importance(model_imps, product_weights)
                if ensemble_imp:
                    result["ensemble"][product_name] = ensemble_imp

    output_path = output_dir / "feature_importance.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Wrote feature importance to {output_path}")
    return output_path


def write_ensemble_fi_with_weights(
    project_name: str,
    all_importances: dict[str, dict[str, dict[str, Any]]],
    model_weights: dict[str, dict[str, float]],
    output_dir: Path | str,
) -> Path:
    """Write enhanced ensemble feature importance JSON with model weights.

    Output structure per the plan:
    {
        "project": "my_project",
        "products": {
            "Product_A": {
                "models": {
                    "XGBoost": {
                        "weight": 45.0,
                        "builtin": {"feature1": 35.0, ...},
                        "shap": {"feature1": 32.0, ...}
                    },
                    "Prophet": {
                        "weight": 35.0,
                        "regressor_coefficients": {"feature1": 30.0, ...}
                    }
                },
                "ensemble": {
                    "importance": {"feature1": 32.0, ...},
                    "method": "inverse_mape_weighted"
                }
            }
        }
    }

    Args:
        project_name: Name of the project.
        all_importances: Nested dict of product -> model -> importance_type -> importance.
        model_weights: Dict of product -> model -> weight for ensemble importance.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    logger = get_logger()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "project": project_name,
        "products": {},
    }

    for product_name, model_data in all_importances.items():
        product_weights = model_weights.get(product_name, {})

        # Build models section with weights (as percentages)
        models_section = {}
        for model_name, imp_data in model_data.items():
            weight = product_weights.get(model_name, 0.0)
            model_entry = {
                "weight": round(float(weight) * 100, 2),  # Convert to percentage
            }
            # Add all importance types (ensure all values are native Python floats)
            for imp_type, imp_values in imp_data.items():
                model_entry[imp_type] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in imp_values.items()
                }

            models_section[model_name] = model_entry

        # Compute ensemble importance
        # Use builtin > shap > permutation > regressor_coefficients priority
        model_imps = {}
        for model_name, imp_data in model_data.items():
            for imp_type in ["builtin", "shap", "permutation", "regressor_coefficients"]:
                if imp_type in imp_data:
                    model_imps[model_name] = imp_data[imp_type]
                    break

        ensemble_imp = {}
        if model_imps and product_weights:
            ensemble_imp = compute_ensemble_importance(model_imps, product_weights)

        result["products"][product_name] = {
            "models": models_section,
            "ensemble": {
                "importance": ensemble_imp,
                "method": "inverse_mape_weighted",
            },
        }

    output_path = output_dir / "feature_importance.json"

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Wrote ensemble feature importance to {output_path}")
    return output_path
