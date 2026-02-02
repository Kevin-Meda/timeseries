"""SARIMA forecasting model implementation."""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecast.models.base import BaseForecaster
from forecast.utils import get_logger


class SARIMAForecaster(BaseForecaster):
    """SARIMA model for time series forecasting.

    This is a univariate model - exogenous features are ignored.
    """

    def __init__(
        self,
        use_optuna: bool = False,
        optuna_trials: int = 50,
        optuna_val_ratio: float = 0.2,
    ):
        """Initialize SARIMA forecaster.

        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization.
            optuna_trials: Number of Optuna trials if optimization is enabled.
            optuna_val_ratio: Ratio of data to use for internal validation during
                Optuna optimization (default 0.2 = 80/20 split).
        """
        super().__init__(name="SARIMA")
        self._supports_multivariate = False
        self._supports_multi_product = False
        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_val_ratio = optuna_val_ratio
        self.order = (1, 1, 1)
        self.seasonal_order = (1, 1, 1, 12)
        self.model = None
        self.fitted_model = None
        self._train_data = None
        self._optuna_trials_used = 0

    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        val: pd.Series | pd.DataFrame | None = None,
        exog_train: pd.DataFrame | None = None,
        exog_val: pd.DataFrame | None = None,
    ) -> None:
        """Fit SARIMA model to training data.

        When use_optuna=True:
        - Combines train+val data
        - Uses internal CV split (optuna_val_ratio) for optimization
        - Test set is never seen during optimization

        Args:
            train: Training time series (Series or DataFrame with 'demand' column).
            val: Optional validation time series for hyperparameter tuning.
            exog_train: Ignored - SARIMA is univariate.
            exog_val: Ignored - SARIMA is univariate.
        """
        # Extract Series if DataFrame is provided
        if isinstance(train, pd.DataFrame):
            train = train["demand"] if "demand" in train.columns else train.iloc[:, 0]
        if isinstance(val, pd.DataFrame):
            val = val["demand"] if "demand" in val.columns else val.iloc[:, 0]
        logger = get_logger()

        # When optimizing, combine train+val and use internal CV
        if self.use_optuna and val is not None:
            combined = pd.concat([train, val])
            self._train_data = combined
            self._optimize_with_optuna(combined)
        elif self.use_optuna:
            # Optimize with internal split from train data only
            self._train_data = train
            self._optimize_with_optuna(train)
        else:
            self._train_data = train
            self._grid_search(train, val)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = SARIMAX(
                    train.values,
                    order=self.order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                self.fitted_model = self.model.fit(disp=False, maxiter=200)
                self._is_fitted = True
                logger.info(
                    f"SARIMA fitted with order={self.order}, "
                    f"seasonal_order={self.seasonal_order}"
                )
        except Exception as e:
            logger.warning(f"SARIMA fitting failed: {e}. Using fallback parameters.")
            self._fit_fallback(train)

    def _grid_search(self, train: pd.Series, val: pd.Series | None) -> None:
        """Perform grid search for optimal SARIMA parameters.

        Args:
            train: Training time series.
            val: Validation time series.
        """
        logger = get_logger()
        logger.debug("Starting SARIMA grid search")

        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]

        best_aic = np.inf
        best_order = self.order
        best_seasonal = self.seasonal_order

        eval_data = val if val is not None else train.iloc[-12:]
        fit_data = train if val is not None else train.iloc[:-12]

        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
                                try:
                                    with warnings.catch_warnings():
                                        warnings.simplefilter("ignore")
                                        model = SARIMAX(
                                            fit_data.values,
                                            order=(p, d, q),
                                            seasonal_order=(P, D, Q, 12),
                                            enforce_stationarity=False,
                                            enforce_invertibility=False,
                                        )
                                        fitted = model.fit(disp=False, maxiter=100)
                                        if fitted.aic < best_aic:
                                            best_aic = fitted.aic
                                            best_order = (p, d, q)
                                            best_seasonal = (P, D, Q, 12)
                                except Exception:
                                    continue

        self.order = best_order
        self.seasonal_order = best_seasonal
        logger.debug(f"Grid search complete: order={best_order}, seasonal={best_seasonal}")

    def _optimize_with_optuna(self, data: pd.Series) -> None:
        """Optimize SARIMA parameters using Optuna with internal CV.

        Uses internal train/val split based on optuna_val_ratio.
        This ensures the test set is never seen during optimization.

        Args:
            data: Combined data for optimization (train+val when available).
        """
        logger = get_logger()

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available, falling back to grid search")
            self._grid_search(data, None)
            return

        # Internal CV split
        split_idx = int(len(data) * (1 - self.optuna_val_ratio))
        internal_train = data.iloc[:split_idx]
        internal_val = data.iloc[split_idx:]

        logger.debug(
            f"SARIMA Optuna internal split: train={len(internal_train)}, "
            f"val={len(internal_val)} (ratio={self.optuna_val_ratio})"
        )

        def objective(trial):
            p = trial.suggest_int("p", 0, 2)
            d = trial.suggest_int("d", 0, 1)
            q = trial.suggest_int("q", 0, 2)
            P = trial.suggest_int("P", 0, 1)
            D = trial.suggest_int("D", 0, 1)
            Q = trial.suggest_int("Q", 0, 1)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = SARIMAX(
                        internal_train.values,
                        order=(p, d, q),
                        seasonal_order=(P, D, Q, 12),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False, maxiter=100)

                    forecast = fitted.forecast(len(internal_val))
                    mape = np.mean(np.abs((internal_val.values - forecast) / (internal_val.values + 1e-8)))
                    return mape
            except Exception:
                return float("inf")

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = study.best_params
        self.order = (best["p"], best["d"], best["q"])
        self.seasonal_order = (best["P"], best["D"], best["Q"], 12)
        self._optuna_trials_used = len(study.trials)
        logger.info(f"Optuna optimization complete: order={self.order}, seasonal={self.seasonal_order}")

    def _fit_fallback(self, train: pd.Series) -> None:
        """Fit with conservative fallback parameters.

        Args:
            train: Training time series.
        """
        logger = get_logger()
        fallback_orders = [
            ((1, 0, 0), (0, 0, 0, 12)),
            ((0, 1, 1), (0, 0, 0, 12)),
            ((1, 1, 0), (0, 0, 0, 12)),
        ]

        for order, seasonal in fallback_orders:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.model = SARIMAX(
                        train.values,
                        order=order,
                        seasonal_order=seasonal,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    self.fitted_model = self.model.fit(disp=False, maxiter=200)
                    self.order = order
                    self.seasonal_order = seasonal
                    self._is_fitted = True
                    logger.info(f"SARIMA fallback successful with order={order}")
                    return
            except Exception:
                continue

        logger.error("All SARIMA fallback attempts failed")
        self._is_fitted = False

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts.

        Args:
            horizon: Number of periods to forecast.
            exog_future: Ignored - SARIMA is univariate.

        Returns:
            Forecasted values with proper DatetimeIndex.
        """
        if not self._is_fitted or self.fitted_model is None:
            raise RuntimeError("Model must be fitted before prediction")

        forecast = self.fitted_model.forecast(horizon)

        # Convert to numpy array if needed
        forecast_values = forecast.values if hasattr(forecast, "values") else forecast

        # Create proper date index from training data
        if self._train_data is not None:
            last_date = self._train_data.index[-1]
            future_index = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=horizon,
                freq="MS",
            )
            return pd.Series(forecast_values, index=future_index, name="SARIMA_forecast")

        return pd.Series(forecast_values, name="SARIMA_forecast")

    def get_params(self) -> dict:
        """Get model parameters for serialization.

        Returns:
            Dictionary of model parameters.
        """
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "use_optuna": self.use_optuna,
            "optuna_trials": self.optuna_trials,
            "optuna_val_ratio": self.optuna_val_ratio,
            "optuna_trials_used": self._optuna_trials_used,
        }

    def load_params(self, params: dict) -> None:
        """Load model parameters.

        Args:
            params: Dictionary of model parameters.
        """
        self.order = tuple(params.get("order", self.order))
        self.seasonal_order = tuple(params.get("seasonal_order", self.seasonal_order))
        self.use_optuna = params.get("use_optuna", self.use_optuna)
        self.optuna_trials = params.get("optuna_trials", self.optuna_trials)
        self.optuna_val_ratio = params.get("optuna_val_ratio", self.optuna_val_ratio)
        self._optuna_trials_used = params.get("optuna_trials_used", 0)

    def save_params(self, path: str) -> None:
        """Save model parameters to JSON file.

        Args:
            path: Path to save parameters.
        """
        params = self.get_params()
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load_params_from_file(self, path: str) -> None:
        """Load model parameters from JSON file.

        Args:
            path: Path to parameter file.
        """
        with open(path, "r") as f:
            params = json.load(f)
        self.load_params(params)
