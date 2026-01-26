"""Preprocessing visualization plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from forecast.utils import get_logger


def plot_preprocessing(
    original: pd.Series,
    cleaned: pd.Series,
    change_log: dict,
    splits: dict,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create preprocessing visualization plot.

    Shows original vs cleaned data with marked replacements and split boundaries.

    Args:
        original: Original time series.
        cleaned: Cleaned time series.
        change_log: Dictionary with replacement information.
        splits: Dictionary with train/val/test split information.
        output_path: Path to save the plot.
        dpi: Figure resolution.
    """
    logger = get_logger()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(original.index, original.values, "b-", label="Original", alpha=0.7, linewidth=1.5)

    neg_indices = change_log.get("negative_indices", [])
    nan_indices = change_log.get("nan_indices", [])
    outlier_indices = change_log.get("outlier_indices", [])
    original_values = change_log.get("original_values", {})

    if neg_indices:
        neg_dates = [original.index[i] for i in neg_indices if i < len(original)]
        neg_vals = [original_values.get(i, original.iloc[i]) for i in neg_indices if i < len(original)]
        ax1.scatter(neg_dates, neg_vals, c="red", marker="v", s=80, label="Negative", zorder=5)

    if nan_indices:
        nan_dates = [original.index[i] for i in nan_indices if i < len(original)]
        ax1.scatter(nan_dates, [0] * len(nan_dates), c="orange", marker="x", s=80, label="NaN", zorder=5)

    if outlier_indices:
        out_dates = [original.index[i] for i in outlier_indices if i < len(original)]
        out_vals = [original_values.get(i, original.iloc[i]) for i in outlier_indices if i < len(original)]
        ax1.scatter(out_dates, out_vals, c="purple", marker="^", s=80, label="Outlier", zorder=5)

    ax1.set_ylabel("Value")
    ax1.set_title(f"Original Data - {original.name}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(cleaned.index, cleaned.values, "g-", label="Cleaned", alpha=0.7, linewidth=1.5)

    if "train_idx" in splits and "val_idx" in splits:
        val_start = splits["val_idx"][0] if len(splits["val_idx"]) > 0 else None
        test_start = splits["test_idx"][0] if len(splits["test_idx"]) > 0 else None

        if val_start is not None:
            ax2.axvline(x=val_start, color="orange", linestyle="--", label="Val Start", alpha=0.8)
        if test_start is not None:
            ax2.axvline(x=test_start, color="red", linestyle="--", label="Test Start", alpha=0.8)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Cleaned Data with Splits - {cleaned.name}")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    logger.debug(f"Preprocessing plot saved: {output_path}")


def plot_all_preprocessing(
    original_dict: dict[str, pd.Series],
    cleaned_dict: dict[str, pd.Series],
    change_logs: dict[str, dict],
    splits_dict: dict[str, dict],
    output_dir: str,
    dpi: int = 150,
) -> None:
    """Create preprocessing plots for all categories.

    Args:
        original_dict: Dictionary of original series.
        cleaned_dict: Dictionary of cleaned series.
        change_logs: Dictionary of change logs.
        splits_dict: Dictionary of split information.
        output_dir: Output directory for plots.
        dpi: Figure resolution.
    """
    logger = get_logger()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name in cleaned_dict.keys():
        if name in original_dict and name in change_logs:
            safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
            plot_path = output_path / f"preprocessing_{safe_name}.png"
            plot_preprocessing(
                original=original_dict[name],
                cleaned=cleaned_dict[name],
                change_log=change_logs[name],
                splits=splits_dict.get(name, {}),
                output_path=str(plot_path),
                dpi=dpi,
            )

    logger.info(f"Preprocessing plots saved to: {output_dir}")
