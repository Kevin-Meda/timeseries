"""Forecast visualization plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from forecast.utils import get_logger


def plot_forecast(
    history: pd.Series,
    forecasts: dict[str, pd.Series],
    ensemble_forecast: pd.Series | None,
    category_name: str,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create forecast visualization with history and future predictions.

    Args:
        history: Historical time series (training data).
        forecasts: Dictionary of model forecasts.
        ensemble_forecast: Ensemble forecast series (optional).
        category_name: Name of the category.
        output_path: Path to save the plot.
        dpi: Figure resolution.
    """
    logger = get_logger()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(history.index, history.values, "k-", label="History", linewidth=2, alpha=0.8)

    colors = plt.cm.Set2(range(len(forecasts)))

    for idx, (model_name, forecast) in enumerate(forecasts.items()):
        ax.plot(forecast.index, forecast.values, color=colors[idx],
                linestyle="--", label=f"{model_name}", linewidth=1.5, alpha=0.7)

    if ensemble_forecast is not None and len(ensemble_forecast) > 0:
        ax.plot(ensemble_forecast.index, ensemble_forecast.values, "r-",
                label="Ensemble", linewidth=2.5, alpha=0.9)

        ax.fill_between(
            ensemble_forecast.index,
            ensemble_forecast.values * 0.9,
            ensemble_forecast.values * 1.1,
            color="red", alpha=0.1, label="Ensemble ±10%"
        )

    if len(history) > 0 and len(forecasts) > 0:
        first_forecast = list(forecasts.values())[0]
        if len(first_forecast) > 0:
            transition_x = history.index[-1]
            ax.axvline(x=transition_x, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"Forecast - {category_name}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Forecast plot saved: {output_path}")


def plot_forecast_detail(
    history: pd.Series,
    forecast: pd.Series,
    model_name: str,
    category_name: str,
    output_path: str,
    confidence_intervals: tuple[pd.Series, pd.Series] | None = None,
    dpi: int = 150,
) -> None:
    """Create detailed forecast plot for a single model.

    Args:
        history: Historical time series.
        forecast: Forecast series.
        model_name: Name of the model.
        category_name: Name of the category.
        output_path: Path to save the plot.
        confidence_intervals: Optional tuple of (lower, upper) confidence bounds.
        dpi: Figure resolution.
    """
    logger = get_logger()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    recent_history = history.iloc[-36:] if len(history) > 36 else history
    ax.plot(recent_history.index, recent_history.values, "b-",
            label="History", linewidth=2, alpha=0.8)

    ax.plot(forecast.index, forecast.values, "r--",
            label=f"{model_name} Forecast", linewidth=2)

    if confidence_intervals is not None:
        lower, upper = confidence_intervals
        ax.fill_between(forecast.index, lower.values, upper.values,
                        color="red", alpha=0.2, label="95% CI")

    if len(recent_history) > 0 and len(forecast) > 0:
        last_hist = recent_history.iloc[-1]
        first_fore = forecast.iloc[0]
        ax.plot([recent_history.index[-1], forecast.index[0]],
                [last_hist, first_fore], "g:", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(f"{model_name} Forecast - {category_name}")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Forecast detail plot saved: {output_path}")


def plot_all_forecasts(
    histories: dict[str, pd.Series],
    forecasts: dict[str, dict[str, pd.Series]],
    ensemble_forecasts: dict[str, pd.Series],
    output_dir: str,
    dpi: int = 150,
) -> None:
    """Create forecast plots for all categories.

    Args:
        histories: Dictionary of historical series.
        forecasts: Nested dict: category -> model -> forecast.
        ensemble_forecasts: Dictionary of ensemble forecasts.
        output_dir: Output directory for plots.
        dpi: Figure resolution.
    """
    logger = get_logger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for category in histories.keys():
        if category in forecasts:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in category)

            combined_path = output_path / f"forecast_{safe_name}.png"
            plot_forecast(
                history=histories[category],
                forecasts=forecasts[category],
                ensemble_forecast=ensemble_forecasts.get(category),
                category_name=category,
                output_path=str(combined_path),
                dpi=dpi,
            )

            for model_name, forecast in forecasts[category].items():
                model_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in model_name)
                detail_path = output_path / f"forecast_{safe_name}_{model_safe}.png"
                plot_forecast_detail(
                    history=histories[category],
                    forecast=forecast,
                    model_name=model_name,
                    category_name=category,
                    output_path=str(detail_path),
                    dpi=dpi,
                )

    logger.info(f"Forecast plots saved to: {output_dir}")
