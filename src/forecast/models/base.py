"""Base class for forecasting models."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models.

    Supports both univariate and multivariate forecasting through optional
    exogenous features.
    """

    def __init__(self, name: str):
        """Initialize the forecaster.

        Args:
            name: Model name for identification.
        """
        self.name = name
        self._is_fitted = False
        self._supports_multivariate = False
        self._supports_multi_product = False

    @abstractmethod
    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        val: pd.Series | pd.DataFrame | None = None,
        exog_train: pd.DataFrame | None = None,
        exog_val: pd.DataFrame | None = None,
    ) -> None:
        """Fit the model to training data.

        Args:
            train: Training time series (Series) or DataFrame with features.
            val: Optional validation time series/DataFrame for early stopping or tuning.
            exog_train: Optional external features for training period.
            exog_val: Optional external features for validation period.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.Series | pd.DataFrame:
        """Generate forecasts for the specified horizon.

        Args:
            horizon: Number of periods to forecast.
            exog_future: Optional external features for forecast period.

        Returns:
            Forecasted values as a pandas Series or DataFrame (for multi-product).
        """
        pass

    @abstractmethod
    def get_params(self) -> dict:
        """Get model parameters for serialization.

        Returns:
            Dictionary of model parameters.
        """
        pass

    @abstractmethod
    def load_params(self, params: dict) -> None:
        """Load model parameters from a dictionary.

        Args:
            params: Dictionary of model parameters.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted

    @property
    def supports_multivariate(self) -> bool:
        """Check if the model supports multivariate input (exogenous features)."""
        return self._supports_multivariate

    @property
    def supports_multi_product(self) -> bool:
        """Check if the model supports multi-product forecasting."""
        return self._supports_multi_product
