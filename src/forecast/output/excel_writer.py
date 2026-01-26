"""Excel output writer for forecasting results."""

from pathlib import Path

import pandas as pd
import numpy as np

from forecast.utils import get_logger


def write_model_results(
    results: dict[str, dict[str, dict[str, float]]],
    path: str,
) -> None:
    """Write model evaluation results to Excel.

    Args:
        results: Nested dict: category -> model -> metrics.
        path: Output file path.
    """
    logger = get_logger()

    if not results:
        logger.warning("No results to write")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for category, model_results in results.items():
        row = {"Category": category}
        for model_name, metrics in model_results.items():
            for metric_name, value in metrics.items():
                col_name = f"{model_name}_{metric_name.upper()}"
                row[col_name] = value
        rows.append(row)

    df = pd.DataFrame(rows)

    aggregate_row = {"Category": "AGGREGATE"}
    numeric_cols = [c for c in df.columns if c != "Category"]
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) > 0:
            aggregate_row[col] = values.mean()
    rows.append(aggregate_row)

    df = pd.DataFrame(rows)

    cols = ["Category"] + sorted([c for c in df.columns if c != "Category"])
    df = df[cols]

    df.to_excel(path, index=False, sheet_name="Model Results")
    logger.info(f"Model results written to: {path}")


def write_ensemble_results(
    ensemble_results: dict[str, dict],
    path: str,
) -> None:
    """Write ensemble forecast results to Excel.

    Args:
        ensemble_results: Dict mapping category to ensemble info:
            - forecast: pd.Series of forecast values
            - mape: ensemble MAPE
            - models_used: list of model names
            - weights: dict of model weights
        path: Output file path.
    """
    logger = get_logger()

    if not ensemble_results:
        logger.warning("No ensemble results to write")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for category, info in ensemble_results.items():
        forecast = info.get("forecast", pd.Series())
        row = {"Category": category}

        for i, val in enumerate(forecast.values):
            row[f"Month_{i+1}"] = val

        row["Ensemble_MAPE"] = info.get("mape", np.nan)
        row["Models_Used"] = ",".join(info.get("models_used", []))

        weights = info.get("weights", {})
        weight_strs = [f"{w:.3f}" for w in weights.values()]
        row["Weights"] = ",".join(weight_strs)

        rows.append(row)

    df = pd.DataFrame(rows)

    cols = ["Category"]
    month_cols = sorted([c for c in df.columns if c.startswith("Month_")],
                        key=lambda x: int(x.split("_")[1]))
    other_cols = ["Ensemble_MAPE", "Models_Used", "Weights"]
    cols = cols + month_cols + [c for c in other_cols if c in df.columns]
    df = df[[c for c in cols if c in df.columns]]

    df.to_excel(path, index=False, sheet_name="Ensemble Forecasts")
    logger.info(f"Ensemble results written to: {path}")


def write_forecast_details(
    forecast_details: dict[str, dict[str, pd.Series]],
    path: str,
) -> None:
    """Write detailed forecasts per model to Excel with multiple sheets.

    Args:
        forecast_details: Dict: category -> model -> forecast Series.
        path: Output file path.
    """
    logger = get_logger()

    if not forecast_details:
        logger.warning("No forecast details to write")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        all_models = set()
        for model_forecasts in forecast_details.values():
            all_models.update(model_forecasts.keys())

        for model_name in sorted(all_models):
            rows = []
            for category, model_forecasts in forecast_details.items():
                if model_name in model_forecasts:
                    forecast = model_forecasts[model_name]
                    row = {"Category": category}
                    for i, (idx, val) in enumerate(forecast.items()):
                        date_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)
                        row[date_str] = val
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                sheet_name = model_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Forecast details written to: {path}")
