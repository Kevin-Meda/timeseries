"""Base class for forecasting models."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseForecaster(ABC):
    """Abstract base class for all forecasting models."""

    def __init__(self, name: str):
        """Initialize the forecaster.

        Args:
            name: Model name for identification.
        """
        self.name = name
        self._is_fitted = False

    @abstractmethod
    def fit(self, train: pd.Series, val: pd.Series | None = None) -> None:
        """Fit the model to training data.

        Args:
            train: Training time series.
            val: Optional validation time series for early stopping or tuning.
        """
        pass

    @abstractmethod
    def predict(self, horizon: int) -> pd.Series:
        """Generate forecasts for the specified horizon.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Forecasted values as a pandas Series.
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
