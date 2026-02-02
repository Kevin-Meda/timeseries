"""Chronos-2 forecasting model wrapper with multivariate support."""

import json

import numpy as np
import pandas as pd

from forecast.models.base import BaseForecaster
from forecast.utils import get_logger

CHRONOS_AVAILABLE = False
try:
    from chronos import Chronos2Pipeline

    CHRONOS_AVAILABLE = True
except ImportError:
    pass


def is_chronos_available() -> bool:
    """Check if Chronos dependencies are available."""
    return CHRONOS_AVAILABLE


class ChronosForecaster(BaseForecaster):
    """Chronos-2 model wrapper for time series forecasting.

    Supports multivariate forecasting with covariates (Chronos-2 feature).
    Covariates are passed as past_covariates during prediction.
    """

    def __init__(self, model_name: str = "amazon/chronos-2"):
        """Initialize Chronos-2 forecaster.

        Args:
            model_name: Hugging Face model name for Chronos-2.
        """
        super().__init__(name="Chronos")
        self._supports_multivariate = True  # Chronos-2 supports covariates
        self._supports_multi_product = True  # Can batch multiple products
        self.model_name = model_name
        self.pipeline = None
        self._train_data = None
        self._context_length = None
        self._covariates = None  # Store covariates for prediction

    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        val: pd.Series | pd.DataFrame | None = None,
        exog_train: pd.DataFrame | None = None,
        exog_val: pd.DataFrame | None = None,
    ) -> None:
        """Initialize Chronos-2 pipeline with training data.

        Args:
            train: Training time series (Series or DataFrame with 'demand' column).
            val: Validation time series (appended to context if provided).
            exog_train: Training exogenous features (used as past_covariates).
            exog_val: Validation exogenous features (appended if provided).
        """
        # Extract Series if DataFrame is provided
        if isinstance(train, pd.DataFrame):
            train = train["demand"] if "demand" in train.columns else train.iloc[:, 0]
        if isinstance(val, pd.DataFrame):
            val = val["demand"] if "demand" in val.columns else val.iloc[:, 0]
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

        # Combine covariates if provided
        self._covariates = None
        if exog_train is not None:
            if exog_val is not None and len(exog_val) > 0:
                self._covariates = pd.concat([exog_train, exog_val])
            else:
                self._covariates = exog_train
            logger.debug(
                f"Chronos-2 using {len(self._covariates.columns)} covariates: "
                f"{list(self._covariates.columns)}"
            )

        self._context_length = len(self._train_data)

        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.pipeline = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=device,
                dtype=torch.float32,
            )
            self._is_fitted = True
            logger.info(
                f"Chronos-2 initialized: model={self.model_name}, "
                f"context_length={self._context_length}, "
                f"covariates={self._covariates is not None}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Chronos-2: {e}")
            self._is_fitted = False

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts using Chronos-2.

        Args:
            horizon: Number of periods to forecast (prediction_length).
            exog_future: Future exogenous features (used as known_covariates).

        Returns:
            Forecasted values with proper DatetimeIndex.
        """
        logger = get_logger()

        if not self._is_fitted or self.pipeline is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self._train_data is None:
            raise RuntimeError("No training data available for context")

        try:
            import torch

            context = torch.tensor(self._train_data.values, dtype=torch.float32)
            logger.debug(
                f"Chronos-2 predict: context_length={len(context)}, "
                f"prediction_length={horizon}"
            )

            # Prepare covariates if available
            past_covariates = None
            known_covariates = None

            if self._covariates is not None:
                # Past covariates: historical feature values
                past_covariates = torch.tensor(
                    self._covariates.values, dtype=torch.float32
                )
                logger.debug(f"Using past_covariates shape: {past_covariates.shape}")

            if exog_future is not None:
                # Known covariates: future feature values
                known_covariates = torch.tensor(
                    exog_future.values, dtype=torch.float32
                )
                logger.debug(f"Using known_covariates shape: {known_covariates.shape}")

            # Chronos-2 API: use predict_quantiles with covariates
            try:
                if past_covariates is not None or known_covariates is not None:
                    # Try multivariate prediction with covariates
                    median_forecasts, _ = self.pipeline.predict_quantiles(
                        inputs=[context],
                        prediction_length=horizon,
                        quantile_levels=[0.5],
                        past_covariates=[past_covariates] if past_covariates is not None else None,
                        known_covariates=[known_covariates] if known_covariates is not None else None,
                    )
                else:
                    # Univariate prediction (no covariates)
                    median_forecasts, _ = self.pipeline.predict_quantiles(
                        inputs=[context],
                        prediction_length=horizon,
                        quantile_levels=[0.5],
                    )
            except TypeError as e:
                # Fallback if covariates not supported by this Chronos version
                logger.warning(
                    f"Covariates not supported by Chronos version, falling back to univariate: {e}"
                )
                median_forecasts, _ = self.pipeline.predict_quantiles(
                    inputs=[context],
                    prediction_length=horizon,
                    quantile_levels=[0.5],
                )

            # median_forecasts is a list of tensors, one per input
            # Each tensor has shape (num_quantiles, prediction_length)
            median_forecast = median_forecasts[0][0].numpy().flatten()

            # Create proper date index
            last_date = self._train_data.index[-1]
            future_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS",
            )

            return pd.Series(median_forecast, index=future_index, name="Chronos_forecast")

        except Exception as e:
            logger.error(f"Chronos-2 prediction failed: {e}")
            raise

    def get_params(self) -> dict:
        """Get model parameters for serialization."""
        return {
            "model_name": self.model_name,
            "context_length": self._context_length,
            "has_covariates": self._covariates is not None,
            "n_covariates": len(self._covariates.columns) if self._covariates is not None else 0,
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
