"""Component decomposition for Prophet models."""

from pathlib import Path
from typing import Any

import pandas as pd

from forecast.utils import get_logger

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


class ProphetDecomposition:
    """Extract and visualize Prophet model components."""

    def __init__(self, model: Any, train_df: pd.DataFrame):
        """Initialize decomposition.

        Args:
            model: Fitted Prophet model.
            train_df: Training DataFrame with 'ds' and 'y' columns.
        """
        self.model = model
        self.train_df = train_df
        self._components: dict[str, pd.DataFrame] = {}
        self._forecast: pd.DataFrame | None = None

    def extract_components(self) -> dict[str, pd.DataFrame]:
        """Extract Prophet components (trend, seasonality, regressors).

        Returns:
            Dictionary with component DataFrames.
        """
        logger = get_logger()

        if self.model is None:
            return {}

        # Create forecast for historical period
        self._forecast = self.model.predict(self.train_df)

        components = {}

        # Trend
        if "trend" in self._forecast.columns:
            components["trend"] = self._forecast[["ds", "trend"]].copy()

        # Yearly seasonality
        if "yearly" in self._forecast.columns:
            components["yearly"] = self._forecast[["ds", "yearly"]].copy()

        # Weekly seasonality (if present)
        if "weekly" in self._forecast.columns:
            components["weekly"] = self._forecast[["ds", "weekly"]].copy()

        # Additive terms (sum of regressors if present)
        if "additive_terms" in self._forecast.columns:
            components["additive_terms"] = self._forecast[["ds", "additive_terms"]].copy()

        # Multiplicative terms
        if "multiplicative_terms" in self._forecast.columns:
            mult_col = self._forecast["multiplicative_terms"]
            if mult_col.abs().sum() > 0:  # Only add if non-zero
                components["multiplicative_terms"] = self._forecast[
                    ["ds", "multiplicative_terms"]
                ].copy()

        # Individual regressor contributions
        for col in self._forecast.columns:
            if col.endswith("_effect"):
                # This is a regressor effect column
                regressor_name = col.replace("_effect", "")
                components[regressor_name] = self._forecast[["ds", col]].copy()
                components[regressor_name].columns = ["ds", regressor_name]

        self._components = components
        logger.debug(f"Extracted Prophet components: {list(components.keys())}")

        return components

    def plot_components(
        self,
        save_dir: str | Path | None = None,
        product_name: str = "",
        figsize: tuple[int, int] = (12, 4),
    ) -> None:
        """Plot Prophet components.

        Args:
            save_dir: Optional directory to save plots.
            product_name: Product name for labeling.
            figsize: Figure size for each component plot.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if not self._components:
            self.extract_components()

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{product_name}_" if product_name else ""

        for name, df in self._components.items():
            plt.figure(figsize=figsize)

            # Get the value column (second column)
            value_col = df.columns[1]

            plt.plot(df["ds"], df[value_col], label=name)
            plt.title(f"Prophet Component: {name}")
            plt.xlabel("Date")
            plt.ylabel(name)
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_dir:
                plt.savefig(
                    save_dir / f"{prefix}component_{name}.png",
                    bbox_inches="tight",
                    dpi=150,
                )
                plt.close()
            else:
                plt.show()

    def plot_all_components(
        self,
        save_path: str | Path | None = None,
        product_name: str = "",
    ) -> None:
        """Plot all components in a single figure.

        Args:
            save_path: Optional path to save the plot.
            product_name: Product name for labeling.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        if not self._components:
            self.extract_components()

        n_components = len(self._components)
        if n_components == 0:
            return

        fig, axes = plt.subplots(n_components, 1, figsize=(12, 3 * n_components))
        if n_components == 1:
            axes = [axes]

        for ax, (name, df) in zip(axes, self._components.items()):
            value_col = df.columns[1]
            ax.plot(df["ds"], df[value_col])
            ax.set_title(f"{name}")
            ax.set_xlabel("Date")
            ax.grid(True, alpha=0.3)

        title = f"Prophet Components - {product_name}" if product_name else "Prophet Components"
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            plt.show()

    def get_component_summary(self) -> dict[str, dict[str, float]]:
        """Get summary statistics for each component.

        Returns:
            Dictionary with component statistics.
        """
        if not self._components:
            self.extract_components()

        summary = {}
        for name, df in self._components.items():
            value_col = df.columns[1]
            values = df[value_col]

            summary[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "range": float(values.max() - values.min()),
            }

        return summary

    @property
    def components(self) -> dict[str, pd.DataFrame]:
        """Get extracted components."""
        if not self._components:
            self.extract_components()
        return self._components

    @property
    def forecast(self) -> pd.DataFrame | None:
        """Get full forecast DataFrame."""
        return self._forecast


def decompose_prophet(
    model: Any,
    train_df: pd.DataFrame,
    save_dir: str | Path | None = None,
    product_name: str = "",
) -> dict[str, Any]:
    """Convenience function to decompose a Prophet model.

    Args:
        model: Fitted Prophet model.
        train_df: Training DataFrame with 'ds' and 'y' columns.
        save_dir: Optional directory to save plots.
        product_name: Product name for labeling.

    Returns:
        Dictionary with component summary and statistics.
    """
    decomp = ProphetDecomposition(model, train_df)
    decomp.extract_components()

    if save_dir:
        decomp.plot_all_components(
            save_path=Path(save_dir) / f"{product_name}_prophet_components.png",
            product_name=product_name,
        )

    return {
        "components": list(decomp.components.keys()),
        "summary": decomp.get_component_summary(),
    }
