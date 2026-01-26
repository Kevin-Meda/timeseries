"""Chronos forecasting model wrapper."""

import json

import numpy as np
import pandas as pd

from forecast.models.base import BaseForecaster
from forecast.utils import get_logger

CHRONOS_AVAILABLE = False
try:
    import torch
    from chronos import ChronosPipeline

    CHRONOS_AVAILABLE = True
except ImportError:
    pass


def is_chronos_available() -> bool:
    """Check if Chronos dependencies are available."""
    return CHRONOS_AVAILABLE


class ChronosForecaster(BaseForecaster):
    """Chronos model wrapper for time series forecasting."""

    def __init__(self, model_name: str = "amazon/chronos-t5-small"):
        """Initialize Chronos forecaster.

        Args:
            model_name: Hugging Face model name for Chronos.
        """
        super().__init__(name="Chronos")
        self.model_name = model_name
        self.pipeline = None
        self._train_data = None
        self._context_length = None

    def fit(self, train: pd.Series, val: pd.Series | None = None) -> None:
        """Initialize Chronos pipeline with training data.

        Args:
            train: Training time series (used as context).
            val: Validation time series (appended to context if provided).
        """
        logger = get_logger()

        if not CHRONOS_AVAILABLE:
            logger.warning(
                "Chronos dependencies not available. "
                "Install with: pip install torch chronos-forecasting"
            )
            self._is_fitted = False
            return

        # Combine train and val for context
        if val is not None and len(val) > 0:
            self._train_data = pd.concat([train, val])
        else:
            self._train_data = train

        self._context_length = len(self._train_data)

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=device,
                dtype=torch.float32,
            )
            self._is_fitted = True
            logger.info(
                f"Chronos initialized: model={self.model_name}, "
                f"context_length={self._context_length}, device={device}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chronos: {e}")
            self._is_fitted = False

    def predict(self, horizon: int) -> pd.Series:
        """Generate forecasts using Chronos.

        Args:
            horizon: Number of periods to forecast (prediction_length).

        Returns:
            Forecasted values with proper DatetimeIndex.
        """
        logger = get_logger()

        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self._train_data is None:
            raise RuntimeError("No training data available for context")

        try:
            inputs = torch.tensor(self._train_data.values, dtype=torch.float32)
            logger.debug(
                f"Chronos predict: context_length={len(inputs)}, "
                f"prediction_length={horizon}"
            )

            forecast = self.pipeline.predict(
                inputs=inputs.unsqueeze(0),
                prediction_length=horizon,
                num_samples=20,
            )

            median_forecast = np.median(forecast[0].numpy(), axis=0)

            # Create proper date index
            last_date = self._train_data.index[-1]
            future_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS",
            )

            return pd.Series(median_forecast, index=future_index, name="Chronos_forecast")

        except Exception as e:
            logger.error(f"Chronos prediction failed: {e}")
            raise

    def get_params(self) -> dict:
        """Get model parameters for serialization."""
        return {
            "model_name": self.model_name,
            "context_length": self._context_length,
        }

    def load_params(self, params: dict) -> None:
        """Load model parameters."""
        self.model_name = params.get("model_name", self.model_name)
        self._context_length = params.get("context_length", self._context_length)

    def save_params(self, path: str) -> None:
        """Save model parameters to JSON file."""
        params = self.get_params()
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load_params_from_file(self, path: str) -> None:
        """Load model parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.load_params(params)
