"""SHAP explainer for XGBoost models."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.utils import get_logger

SHAP_AVAILABLE = False
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    pass

MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    pass


def is_shap_available() -> bool:
    """Check if SHAP is available."""
    return SHAP_AVAILABLE


class ShapExplainer:
    """SHAP explainer for tree-based models (XGBoost)."""

    def __init__(self, model: Any, feature_names: list[str] | None = None):
        """Initialize SHAP explainer.

        Args:
            model: Fitted XGBoost model or any tree-based model.
            feature_names: List of feature column names.
        """
        self.model = model
        self.feature_names = feature_names
        self._explainer: Any = None
        self._shap_values: np.ndarray | None = None
        self._X: pd.DataFrame | None = None

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for given data.

        Args:
            X: Feature DataFrame.

        Returns:
            SHAP values array.

        Raises:
            ImportError: If SHAP is not available.
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        logger = get_logger()

        self._X = X
        if self.feature_names is None:
            self.feature_names = list(X.columns)

        # Create TreeExplainer for XGBoost
        self._explainer = shap.TreeExplainer(self.model)
        self._shap_values = self._explainer.shap_values(X)

        logger.debug(f"Computed SHAP values for {len(X)} samples, {len(self.feature_names)} features")

        return self._shap_values

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance based on mean absolute SHAP values.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if self._shap_values is None or self.feature_names is None:
            return {}

        # Mean absolute SHAP value per feature
        importance = np.abs(self._shap_values).mean(axis=0)

        return dict(zip(self.feature_names, importance))

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples sorted by importance.
        """
        importance = self.get_feature_importance()
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    def plot_summary(
        self,
        save_path: str | Path | None = None,
        max_display: int = 20,
    ) -> None:
        """Generate SHAP summary plot.

        Args:
            save_path: Optional path to save the plot.
            max_display: Maximum number of features to display.
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return

        if self._shap_values is None or self._X is None:
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            plt.show()

    def plot_bar(
        self,
        save_path: str | Path | None = None,
        max_display: int = 20,
    ) -> None:
        """Generate SHAP bar plot (feature importance).

        Args:
            save_path: Optional path to save the plot.
            max_display: Maximum number of features to display.
        """
        if not SHAP_AVAILABLE or not MATPLOTLIB_AVAILABLE:
            return

        if self._shap_values is None or self._X is None:
            return

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self._shap_values,
            self._X,
            feature_names=self.feature_names,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            plt.show()

    def to_dict(self) -> dict[str, Any]:
        """Convert SHAP analysis results to dictionary.

        Returns:
            Dictionary with feature importance and metadata.
        """
        return {
            "feature_importance": self.get_feature_importance(),
            "top_features": self.get_top_features(10),
            "n_samples": len(self._X) if self._X is not None else 0,
            "n_features": len(self.feature_names) if self.feature_names else 0,
        }


def explain_xgboost(
    model: Any,
    X: pd.DataFrame,
    save_dir: str | Path | None = None,
    product_name: str = "",
) -> dict[str, Any]:
    """Convenience function to explain an XGBoost model.

    Args:
        model: Fitted XGBoost model.
        X: Feature DataFrame used for training.
        save_dir: Optional directory to save plots.
        product_name: Product name for labeling.

    Returns:
        Dictionary with feature importance results.
    """
    if not SHAP_AVAILABLE:
        return {"error": "SHAP not available"}

    explainer = ShapExplainer(model, list(X.columns))
    explainer.compute_shap_values(X)

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{product_name}_" if product_name else ""
        explainer.plot_summary(save_dir / f"{prefix}shap_summary.png")
        explainer.plot_bar(save_dir / f"{prefix}shap_bar.png")

    return explainer.to_dict()
