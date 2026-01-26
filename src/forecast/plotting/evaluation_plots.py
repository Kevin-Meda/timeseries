"""Evaluation visualization plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from forecast.utils import get_logger


def plot_evaluation(
    actual: pd.Series,
    predictions: dict[str, pd.Series],
    metrics: dict[str, dict[str, float]],
    category_name: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create evaluation plot comparing actual vs predicted values.

    Args:
        actual: Actual test values.
        predictions: Dictionary of model predictions.
        metrics: Dictionary of model metrics.
        category_name: Name of the category.
        output_path: Path to save the plot.
        dpi: Figure resolution.
    """
    logger = get_logger()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n_models = len(predictions)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 4 * n_models), squeeze=False)

    colors = plt.cm.Set1(range(n_models))

    for idx, (model_name, pred) in enumerate(predictions.items()):
        ax = axes[idx, 0]

        ax.plot(actual.index, actual.values, "k-", label="Actual", linewidth=2, alpha=0.8)
        ax.plot(pred.index if hasattr(pred, "index") else actual.index,
                pred.values, color=colors[idx], linestyle="--",
                label=f"{model_name} Prediction", linewidth=2)

        model_metrics = metrics.get(model_name, {})
        rmse = model_metrics.get("rmse", "N/A")
        mape = model_metrics.get("mape", "N/A")
        mae = model_metrics.get("mae", "N/A")

        if isinstance(mape, float):
            mape_str = f"{mape:.2%}"
        else:
            mape_str = str(mape)

        title = f"{category_name} - {model_name}"
        subtitle = f"RMSE: {rmse:.2f}, MAPE: {mape_str}, MAE: {mae:.2f}" if isinstance(rmse, float) else ""
        ax.set_title(f"{title}\n{subtitle}", fontsize=11)

        ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    axes[-1, 0].set_xlabel("Date")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Evaluation plot saved: {output_path}")


def plot_model_comparison(
    actual: pd.Series,
    predictions: dict[str, pd.Series],
    metrics: dict[str, dict[str, float]],
    category_name: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create a single plot comparing all models.

    Args:
        actual: Actual test values.
        predictions: Dictionary of model predictions.
        metrics: Dictionary of model metrics.
        category_name: Name of the category.
        output_path: Path to save the plot.
        dpi: Figure resolution.
    """
    logger = get_logger()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(actual.index, actual.values, "k-", label="Actual", linewidth=2.5, alpha=0.9)

    colors = plt.cm.Set1(range(len(predictions)))

    for idx, (model_name, pred) in enumerate(predictions.items()):
        model_metrics = metrics.get(model_name, {})
        mape = model_metrics.get("mape", 0)
        mape_str = f"{mape:.1%}" if isinstance(mape, float) else "N/A"

        label = f"{model_name} (MAPE: {mape_str})"
        ax.plot(pred.index if hasattr(pred, "index") else actual.index,
                pred.values, color=colors[idx], linestyle="--",
                label=label, linewidth=1.8, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Model Comparison - {category_name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Model comparison plot saved: {output_path}")


def plot_all_evaluations(
    actuals: dict[str, pd.Series],
    predictions: dict[str, dict[str, pd.Series]],
    metrics: dict[str, dict[str, dict[str, float]]],
    output_dir: str,
    dpi: int = 150,
) -> None:
    """Create evaluation plots for all categories.

    Args:
        actuals: Dictionary of actual test series.
        predictions: Nested dict: category -> model -> predictions.
        metrics: Nested dict: category -> model -> metrics.
        output_dir: Output directory for plots.
        dpi: Figure resolution.
    """
    logger = get_logger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for category in actuals.keys():
        if category in predictions:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in category)

            plot_path = output_path / f"evaluation_{safe_name}.png"
            plot_evaluation(
                actual=actuals[category],
                predictions=predictions[category],
                metrics=metrics.get(category, {}),
                category_name=category,
                output_path=str(plot_path),
                dpi=dpi,
            )

            comparison_path = output_path / f"comparison_{safe_name}.png"
            plot_model_comparison(
                actual=actuals[category],
                predictions=predictions[category],
                metrics=metrics.get(category, {}),
                category_name=category,
                output_path=str(comparison_path),
                dpi=dpi,
            )

    logger.info(f"Evaluation plots saved to: {output_dir}")
