"""Excel output writer for forecasting results."""

from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from forecast.utils import get_logger


def get_timestamp() -> str:
    """Get current timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def write_model_results(
    results: dict[str, dict[str, dict[str, float]]],
    test_actuals: dict[str, pd.Series],
    test_predictions: dict[str, dict[str, pd.Series]],
    path: str,
) -> None:
    """Write model evaluation results to Excel with monthly details.

    Args:
        results: Nested dict: category -> model -> metrics.
        test_actuals: Dict of actual test values per category.
        test_predictions: Dict: category -> model -> predictions.
        path: Output file path.
    """
    logger = get_logger()

    if not results:
        logger.warning("No results to write")
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # Summary sheet
        summary_rows = []
        for category, model_results in results.items():
            row = {"Category": category}
            for model_name, metrics in model_results.items():
                for metric_name, value in metrics.items():
                    col_name = f"{model_name}_{metric_name.upper()}"
                    row[col_name] = value
            summary_rows.append(row)

        # Add aggregate row
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            aggregate_row = {"Category": "AGGREGATE"}
            numeric_cols = [c for c in df_summary.columns if c != "Category"]
            for col in numeric_cols:
                values = df_summary[col].dropna()
                if len(values) > 0:
                    aggregate_row[col] = values.mean()
            summary_rows.append(aggregate_row)
            df_summary = pd.DataFrame(summary_rows)

            cols = ["Category"] + sorted([c for c in df_summary.columns if c != "Category"])
            df_summary = df_summary[cols]
            df_summary.to_excel(writer, sheet_name="Summary", index=False)

        # Monthly details sheet per category
        for category in test_actuals.keys():
            if category not in test_predictions:
                continue

            actual = test_actuals[category]
            predictions = test_predictions[category]

            monthly_rows = []
            for i, (date, actual_val) in enumerate(actual.items()):
                date_str = date.strftime("%Y-%m") if hasattr(date, "strftime") else str(date)
                row = {"Month": date_str, "Actual": actual_val}

                for model_name, pred in predictions.items():
                    if i < len(pred):
                        pred_val = pred.iloc[i] if hasattr(pred, "iloc") else pred[i]
                        row[f"{model_name}_Pred"] = pred_val
                        # Calculate monthly MAPE
                        if actual_val != 0:
                            mape = abs((actual_val - pred_val) / actual_val)
                            row[f"{model_name}_MAPE"] = mape

                monthly_rows.append(row)

            if monthly_rows:
                df_monthly = pd.DataFrame(monthly_rows)
                # Ensure proper column order
                cols = ["Month", "Actual"]
                for model_name in predictions.keys():
                    cols.extend([f"{model_name}_Pred", f"{model_name}_MAPE"])
                df_monthly = df_monthly[[c for c in cols if c in df_monthly.columns]]

                safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in category)[:28]
                df_monthly.to_excel(writer, sheet_name=safe_name, index=False)

    logger.info(f"Model results written to: {path}")


def write_ensemble_results(
    ensemble_results: dict[str, dict],
    path: str,
) -> None:
    """Write ensemble forecast results to Excel.

    Args:
        ensemble_results: Dict mapping category to ensemble info.
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

        # Add forecast values with actual dates
        for idx, val in forecast.items():
            date_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else f"Month_{idx}"
            row[date_str] = val

        row["Ensemble_MAPE"] = info.get("mape", np.nan)
        row["Models_Used"] = ",".join(info.get("models_used", []))

        weights = info.get("weights", {})
        weight_strs = [f"{k}:{v:.3f}" for k, v in weights.items()]
        row["Weights"] = ", ".join(weight_strs)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns
    cols = ["Category"]
    date_cols = [c for c in df.columns if c not in ["Category", "Ensemble_MAPE", "Models_Used", "Weights"]]
    other_cols = ["Ensemble_MAPE", "Models_Used", "Weights"]
    cols = cols + date_cols + [c for c in other_cols if c in df.columns]
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
                    for idx, val in forecast.items():
                        date_str = idx.strftime("%Y-%m") if hasattr(idx, "strftime") else str(idx)
                        row[date_str] = val
                    rows.append(row)

            if rows:
                df = pd.DataFrame(rows)
                sheet_name = model_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    logger.info(f"Forecast details written to: {path}")
