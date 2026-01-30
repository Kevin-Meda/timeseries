"""Consolidated JSON writer for run summaries."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from forecast.utils import get_logger


def write_run_summary(
    project_name: str,
    timestamp: str,
    results_dir: Path,
    products_data: dict[str, dict[str, Any]],
    run_config: dict[str, Any] | None = None,
) -> Path:
    """Write consolidated run summary to JSON.

    Output structure:
    {
      "run_info": {
        "project_name": "...",
        "timestamp": "...",
        "run_date": "..."
      },
      "config": {...},
      "products": {
        "Product_A": {
          "models": {
            "XGBoost": {
              "metrics": {"backtest": {...}, "validation": {...}},
              "params": {...},
              "retrained": true,
              "optuna_trials_used": 100,
              "feature_importance": {...}
            }
          }
        }
      }
    }

    Args:
        project_name: Name of the project.
        timestamp: Run timestamp.
        results_dir: Directory to save the JSON file.
        products_data: Dictionary with product results.
        run_config: Optional configuration dictionary.

    Returns:
        Path to the saved JSON file.
    """
    logger = get_logger()

    summary = {
        "run_info": {
            "project_name": project_name,
            "timestamp": timestamp,
            "run_date": datetime.now().isoformat(),
        },
        "config": run_config or {},
        "products": products_data,
    }

    output_path = results_dir / "run_summary.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializer)

    logger.info(f"Wrote run summary to {output_path}")
    return output_path


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for non-standard types.

    Args:
        obj: Object to serialize.

    Returns:
        JSON-serializable representation.
    """
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "__dict__"):
        return str(obj)
    return str(obj)


def build_product_result(
    product_name: str,
    model_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build a product result dictionary for JSON output.

    Args:
        product_name: Name of the product.
        model_results: Dictionary with model-specific results.

    Returns:
        Formatted product result dictionary.
    """
    return {
        "models": model_results,
    }


def build_model_result(
    metrics: dict[str, Any],
    params: dict[str, Any],
    retrained: bool,
    optuna_trials_used: int = 0,
    feature_importance: dict[str, float] | None = None,
    validation_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model result dictionary for JSON output.

    Args:
        metrics: Backtest/test metrics dictionary.
        params: Model parameters.
        retrained: Whether the model was retrained.
        optuna_trials_used: Number of Optuna trials used.
        feature_importance: Optional feature importance dictionary.
        validation_metrics: Optional validation metrics.

    Returns:
        Formatted model result dictionary.
    """
    result = {
        "metrics": {
            "backtest": metrics,
        },
        "params": params,
        "retrained": retrained,
        "optuna_trials_used": optuna_trials_used,
    }

    if validation_metrics:
        result["metrics"]["validation"] = validation_metrics

    if feature_importance:
        result["feature_importance"] = feature_importance

    return result


def load_run_summary(results_dir: Path) -> dict[str, Any] | None:
    """Load a run summary from JSON.

    Args:
        results_dir: Directory containing the run_summary.json file.

    Returns:
        Summary dictionary or None if not found.
    """
    summary_path = results_dir / "run_summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        return json.load(f)
