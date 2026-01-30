"""Prophet forecasting model implementation."""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.models.base import BaseForecaster
from forecast.utils import get_logger

PROPHET_AVAILABLE = False
try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:
    pass


def is_prophet_available() -> bool:
    """Check if Prophet is available."""
    return PROPHET_AVAILABLE


class ProphetForecaster(BaseForecaster):
    """Prophet model for time series forecasting.

    Supports multivariate input with exogenous regressors.
    Handles seasonality and trend automatically.
    """

    def __init__(
        self,
        use_optuna: bool = True,
        optuna_trials: int = 50,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        seasonality_mode: str = "additive",
    ):
        """Initialize Prophet forecaster.

        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization.
            optuna_trials: Number of Optuna trials.
            changepoint_prior_scale: Flexibility of trend changes.
            seasonality_prior_scale: Flexibility of seasonality.
            seasonality_mode: "additive" or "multiplicative".
        """
        super().__init__(name="Prophet")
        self._supports_multivariate = True
        self._supports_multi_product = True

        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.seasonality_mode = seasonality_mode

        self.model: Any = None
        self._train_data: pd.DataFrame | None = None
        self._regressor_columns: list[str] = []
        self._target_col = "demand"
        self._optuna_trials_used = 0
        self._last_regressor_values: dict[str, Any] = {}

    def _prepare_prophet_df(
        self,
        df: pd.DataFrame,
        include_regressors: bool = True,
    ) -> pd.DataFrame:
        """Prepare DataFrame in Prophet format (ds, y, regressors).

        Args:
            df: DataFrame with demand and optional features.
            include_regressors: Whether to include regressor columns.

        Returns:
            DataFrame in Prophet format.
        """
        prophet_df = pd.DataFrame()
        prophet_df["ds"] = df.index
        prophet_df["y"] = df[self._target_col].values

        if include_regressors and self._regressor_columns:
            for col in self._regressor_columns:
                if col in df.columns:
                    prophet_df[col] = df[col].values

        return prophet_df

    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        val: pd.Series | pd.DataFrame | None = None,
        exog_train: pd.DataFrame | None = None,
        exog_val: pd.DataFrame | None = None,
    ) -> None:
        """Fit Prophet model.

        Args:
            train: Training data (Series or DataFrame with 'demand' column).
            val: Optional validation data for tuning.
            exog_train: Optional exogenous features for training period.
            exog_val: Optional exogenous features for validation period.
        """
        logger = get_logger()

        if not PROPHET_AVAILABLE:
            logger.warning("Prophet not available. Install with: pip install prophet")
            self._is_fitted = False
            return

        # Convert Series to DataFrame
        if isinstance(train, pd.Series):
            train_df = pd.DataFrame({self._target_col: train})
        else:
            train_df = train.copy()
            if self._target_col not in train_df.columns:
                train_df[self._target_col] = train_df.iloc[:, 0]

        # Add exogenous features if provided
        if exog_train is not None:
            for col in exog_train.columns:
                if col not in train_df.columns and col != self._target_col:
                    train_df[col] = exog_train[col]

        # Store regressor columns (all columns except target)
        self._regressor_columns = [
            col for col in train_df.columns if col != self._target_col
        ]

        # Log regressor info
        if self._regressor_columns:
            logger.info(
                f"Prophet regressors: {len(self._regressor_columns)} regressors "
                f"({', '.join(self._regressor_columns)})"
            )

        # Store last known regressor values for extrapolation
        if self._regressor_columns:
            self._last_regressor_values = {
                col: train_df[col].iloc[-1] for col in self._regressor_columns
            }

        # Store original data
        self._train_data = train_df.copy()

        # Handle validation data
        val_df = None
        if val is not None:
            if isinstance(val, pd.Series):
                val_df = pd.DataFrame({self._target_col: val})
            else:
                val_df = val.copy()
                if self._target_col not in val_df.columns:
                    val_df[self._target_col] = val_df.iloc[:, 0]

            if exog_val is not None:
                for col in exog_val.columns:
                    if col not in val_df.columns and col != self._target_col:
                        val_df[col] = exog_val[col]

            # Update last regressor values
            if self._regressor_columns:
                self._last_regressor_values = {
                    col: val_df[col].iloc[-1]
                    for col in self._regressor_columns
                    if col in val_df.columns
                }

        # Hyperparameter optimization or direct fitting
        if self.use_optuna and val_df is not None:
            self._optimize_with_optuna(train_df, val_df)
        else:
            self._fit_model(train_df)

        self._is_fitted = True
        logger.info(
            f"Prophet fitted with changepoint_prior_scale={self.changepoint_prior_scale}, "
            f"seasonality_prior_scale={self.seasonality_prior_scale}"
        )

    def _fit_model(self, train_df: pd.DataFrame) -> None:
        """Fit Prophet model with current parameters.

        Args:
            train_df: Training DataFrame.
        """
        prophet_df = self._prepare_prophet_df(train_df)

        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            seasonality_mode=self.seasonality_mode,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
        )

        # Add regressors
        for col in self._regressor_columns:
            if col in prophet_df.columns:
                self.model.add_regressor(col)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(prophet_df)

    def _optimize_with_optuna(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> None:
        """Optimize hyperparameters using Optuna.

        Args:
            train_df: Training DataFrame.
            val_df: Validation DataFrame.
        """
        logger = get_logger()

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available, using default parameters")
            self._fit_model(train_df)
            return

        def objective(trial):
            params = {
                "changepoint_prior_scale": trial.suggest_float(
                    "changepoint_prior_scale", 0.001, 0.5, log=True
                ),
                "seasonality_prior_scale": trial.suggest_float(
                    "seasonality_prior_scale", 0.1, 50.0, log=True
                ),
                "seasonality_mode": trial.suggest_categorical(
                    "seasonality_mode", ["additive", "multiplicative"]
                ),
            }

            prophet_train = self._prepare_prophet_df(train_df)

            model = Prophet(
                changepoint_prior_scale=params["changepoint_prior_scale"],
                seasonality_prior_scale=params["seasonality_prior_scale"],
                seasonality_mode=params["seasonality_mode"],
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
            )

            for col in self._regressor_columns:
                if col in prophet_train.columns:
                    model.add_regressor(col)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_train)

            # Prepare validation data for prediction
            prophet_val = self._prepare_prophet_df(val_df, include_regressors=True)
            future = prophet_val[["ds"] + self._regressor_columns].copy()

            forecast = model.predict(future)
            predictions = forecast["yhat"].values
            actuals = val_df[self._target_col].values

            mape = np.mean(np.abs((actuals - predictions) / (actuals + 1e-8)))
            return mape

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = study.best_params
        self.changepoint_prior_scale = best["changepoint_prior_scale"]
        self.seasonality_prior_scale = best["seasonality_prior_scale"]
        self.seasonality_mode = best["seasonality_mode"]
        self._optuna_trials_used = len(study.trials)

        # Fit final model with best params
        self._fit_model(train_df)

        logger.info(f"Prophet Optuna optimization complete: {best}")

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts.

        Args:
            horizon: Number of periods to forecast.
            exog_future: Optional exogenous features for forecast period.
                If not provided, uses last known values.

        Returns:
            Forecasted values with proper DatetimeIndex.
        """
        logger = get_logger()

        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self._train_data is None:
            raise RuntimeError("No training data available")

        # Create future DataFrame
        last_date = self._train_data.index[-1]
        future_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        future = pd.DataFrame({"ds": future_index})

        # Add regressor values for future periods
        for col in self._regressor_columns:
            if exog_future is not None and col in exog_future.columns:
                # Use provided future values
                future[col] = exog_future[col].values[:horizon]
            elif col in self._last_regressor_values:
                # Extrapolate using last known value
                future[col] = self._last_regressor_values[col]

        forecast = self.model.predict(future)

        return pd.Series(
            forecast["yhat"].values,
            index=future_index,
            name="Prophet_forecast",
        )

    def get_components(self) -> dict[str, pd.DataFrame]:
        """Get Prophet components (trend, seasonality, regressors).

        Returns:
            Dictionary with component DataFrames.
        """
        if not self._is_fitted or self.model is None or self._train_data is None:
            return {}

        prophet_df = self._prepare_prophet_df(self._train_data)
        forecast = self.model.predict(prophet_df)

        components = {
            "trend": forecast[["ds", "trend"]].copy(),
            "yearly": forecast[["ds", "yearly"]].copy() if "yearly" in forecast else None,
        }

        # Add regressor components
        for col in self._regressor_columns:
            if col in forecast.columns:
                components[col] = forecast[["ds", col]].copy()

        return {k: v for k, v in components.items() if v is not None}

    def get_params(self) -> dict:
        """Get model parameters for serialization."""
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "seasonality_mode": self.seasonality_mode,
            "use_optuna": self.use_optuna,
            "optuna_trials": self.optuna_trials,
            "regressor_columns": self._regressor_columns,
            "last_regressor_values": self._last_regressor_values,
            "optuna_trials_used": self._optuna_trials_used,
        }

    def load_params(self, params: dict) -> None:
        """Load model parameters."""
        self.changepoint_prior_scale = params.get(
            "changepoint_prior_scale", self.changepoint_prior_scale
        )
        self.seasonality_prior_scale = params.get(
            "seasonality_prior_scale", self.seasonality_prior_scale
        )
        self.seasonality_mode = params.get("seasonality_mode", self.seasonality_mode)
        self.use_optuna = params.get("use_optuna", self.use_optuna)
        self.optuna_trials = params.get("optuna_trials", self.optuna_trials)
        self._regressor_columns = params.get("regressor_columns", [])
        self._last_regressor_values = params.get("last_regressor_values", {})
        self._optuna_trials_used = params.get("optuna_trials_used", 0)

    def save_params(self, path: str) -> None:
        """Save model parameters to JSON file."""
        params = self.get_params()
        with open(path, "w") as f:
            json.dump(params, f, indent=2, default=str)

    def load_params_from_file(self, path: str) -> None:
        """Load model parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.load_params(params)
