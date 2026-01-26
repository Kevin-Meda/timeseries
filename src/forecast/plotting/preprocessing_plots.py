"""Preprocessing visualization plots."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from forecast.utils import get_logger


def plot_preprocessing(
    original: pd.Series,
    cleaned: pd.Series,
    change_log: dict,
    splits: dict,
    output_path: str,
    dpi: int = 150,
) -> None:
    """Create preprocessing visualization plot."""
    logger = get_logger()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Use integer positions for x-axis, then set custom labels
    x_positions = range(len(original))
    dates = original.index

    # Plot original data
    ax1 = axes[0]
    ax1.plot(x_positions, original.values, "b-", label="Original", alpha=0.7, linewidth=1.5)

    # Mark anomalies
    neg_indices = change_log.get("negative_indices", [])
    nan_indices = change_log.get("nan_indices", [])
    outlier_indices = change_log.get("outlier_indices", [])
    original_values = change_log.get("original_values", {})

    if neg_indices:
        neg_x = [i for i in neg_indices if i < len(original)]
        neg_vals = [original_values.get(i, original.iloc[i]) for i in neg_indices if i < len(original)]
        ax1.scatter(neg_x, neg_vals, c="red", marker="v", s=80, label="Negative", zorder=5)

    if nan_indices:
        nan_x = [i for i in nan_indices if i < len(original)]
        ax1.scatter(nan_x, [0] * len(nan_x), c="orange", marker="x", s=80, label="NaN", zorder=5)

    if outlier_indices:
        out_x = [i for i in outlier_indices if i < len(original)]
        out_vals = [original_values.get(i, original.iloc[i]) for i in outlier_indices if i < len(original)]
        ax1.scatter(out_x, out_vals, c="purple", marker="^", s=80, label="Outlier", zorder=5)

    ax1.set_ylabel("Value")
    ax1.set_title(f"Original Data - {original.name}")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Plot cleaned data
    ax2 = axes[1]
    ax2.plot(x_positions, cleaned.values, "g-", label="Cleaned", alpha=0.7, linewidth=1.5)

    # Add split lines using index positions
    if "val_idx" in splits and len(splits["val_idx"]) > 0:
        val_date = splits["val_idx"][0]
        try:
            val_pos = list(dates).index(val_date)
            ax2.axvline(x=val_pos, color="orange", linestyle="--", label="Val Start", alpha=0.8)
        except (ValueError, IndexError):
            pass

    if "test_idx" in splits and len(splits["test_idx"]) > 0:
        test_date = splits["test_idx"][0]
        try:
            test_pos = list(dates).index(test_date)
            ax2.axvline(x=test_pos, color="red", linestyle="--", label="Test Start", alpha=0.8)
        except (ValueError, IndexError):
            pass

    ax2.set_xlabel("Date")
    ax2.set_ylabel("Value")
    ax2.set_title(f"Cleaned Data with Splits - {cleaned.name}")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    # Set up x-axis ticks for both axes
    tick_interval = max(1, len(dates) // 10)  # Show ~10 ticks
    tick_positions = list(range(0, len(dates), tick_interval))
    tick_labels = [dates[i].strftime("%Y-%m") for i in tick_positions]

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_xlim(-1, len(dates))

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
    """Create preprocessing plots for all categories."""
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
