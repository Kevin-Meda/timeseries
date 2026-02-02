"""XGBoost forecasting model implementation."""

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from forecast.models.base import BaseForecaster
from forecast.features.feature_engineering import (
    create_lag_features,
    create_rolling_features,
    create_calendar_features,
    get_feature_columns,
)
from forecast.utils import get_logger

XGBOOST_AVAILABLE = False
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    pass


def is_xgboost_available() -> bool:
    """Check if XGBoost is available."""
    return XGBOOST_AVAILABLE


class XGBoostForecaster(BaseForecaster):
    """XGBoost model for time series forecasting.

    Supports multivariate input with exogenous features.
    Auto-generates temporal features (lags, rolling, calendar).
    """

    def __init__(
        self,
        use_optuna: bool = True,
        optuna_trials: int = 100,
        optuna_val_ratio: float = 0.2,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        lags: list[int] | None = None,
        rolling_windows: list[int] | None = None,
    ):
        """Initialize XGBoost forecaster.

        Args:
            use_optuna: Whether to use Optuna for hyperparameter optimization.
            optuna_trials: Number of Optuna trials.
            optuna_val_ratio: Ratio of data to use for internal validation during
                Optuna optimization (default 0.2 = 80/20 split).
            n_estimators: Number of boosting rounds (default if not tuning).
            max_depth: Maximum tree depth (default if not tuning).
            learning_rate: Learning rate (default if not tuning).
            lags: Lag periods for feature generation. Defaults to [1,2,3,6,12].
            rolling_windows: Rolling window sizes. Defaults to [3,6,12].
        """
        super().__init__(name="XGBoost")
        self._supports_multivariate = True
        self._supports_multi_product = True

        self.use_optuna = use_optuna
        self.optuna_trials = optuna_trials
        self.optuna_val_ratio = optuna_val_ratio
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.lags = lags or [1, 2, 3, 6, 12]
        self.rolling_windows = rolling_windows or [3, 6, 12]

        self.model: Any = None
        self._train_data: pd.DataFrame | None = None
        self._feature_columns: list[str] = []
        self._target_col = "demand"
        self._optuna_trials_used = 0

    def _prepare_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> pd.DataFrame:
        """Prepare features for XGBoost.

        Args:
            df: DataFrame with demand and optional exogenous features.
            is_training: If True, drops NaN rows from lagging.

        Returns:
            DataFrame with all features.
        """
        result = df.copy()

        # Add temporal features
        if self._target_col in result.columns:
            result = create_lag_features(result, self._target_col, self.lags)
            result = create_rolling_features(result, self._target_col, self.rolling_windows)

        result = create_calendar_features(result)

        if is_training:
            # Drop rows with NaN from lagging (at the beginning)
            result = result.dropna()

        return result

    def fit(
        self,
        train: pd.Series | pd.DataFrame,
        val: pd.Series | pd.DataFrame | None = None,
        exog_train: pd.DataFrame | None = None,
        exog_val: pd.DataFrame | None = None,
    ) -> None:
        """Fit XGBoost model.

        When use_optuna=True:
        - Combines train+val data
        - Uses internal CV split (optuna_val_ratio) for optimization
        - Test set is never seen during optimization

        Args:
            train: Training data (Series or DataFrame with 'demand' column).
            val: Optional validation data for early stopping/tuning.
            exog_train: Optional exogenous features for training period.
            exog_val: Optional exogenous features for validation period.
        """
        logger = get_logger()

        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Install with: pip install xgboost")
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
            exog_cols = [col for col in exog_train.columns if col not in train_df.columns]
            for col in exog_cols:
                train_df[col] = exog_train[col]

            # Log exogenous feature info
            if exog_cols:
                logger.info(
                    f"XGBoost exogenous features: {len(exog_cols)} features "
                    f"({', '.join(exog_cols)})"
                )

        # Handle validation data - combine with train when optimizing
        combined_df = train_df.copy()
        if val is not None:
            if isinstance(val, pd.Series):
                val_df = pd.DataFrame({self._target_col: val})
            else:
                val_df = val.copy()
                if self._target_col not in val_df.columns:
                    val_df[self._target_col] = val_df.iloc[:, 0]

            if exog_val is not None:
                for col in exog_val.columns:
                    if col not in val_df.columns:
                        val_df[col] = exog_val[col]

            combined_df = pd.concat([train_df, val_df])

        # Store combined data for prediction
        self._train_data = combined_df.copy()

        # Prepare features on combined data
        all_features = self._prepare_features(combined_df, is_training=True)

        # Get feature columns (exclude target)
        self._feature_columns = get_feature_columns(all_features, self._target_col)

        # Log total feature count
        logger.info(
            f"XGBoost total features: {len(self._feature_columns)} "
            f"(temporal + exogenous combined)"
        )

        X_all = all_features[self._feature_columns]
        y_all = all_features[self._target_col]

        # Hyperparameter optimization with internal CV
        if self.use_optuna:
            self._optimize_with_optuna(X_all, y_all)
        else:
            self._fit_model(X_all, y_all, None, None)

        self._is_fitted = True
        logger.info(
            f"XGBoost fitted with n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, learning_rate={self.learning_rate}"
        )

    def _fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit the XGBoost model with current parameters.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Optional validation features.
            y_val: Optional validation targets.
        """
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=False,
            )

    def _optimize_with_optuna(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> None:
        """Optimize hyperparameters using Optuna with internal CV.

        Uses internal train/val split based on optuna_val_ratio.
        This ensures the test set is never seen during optimization.

        Args:
            X: All features (combined train+val).
            y: All targets (combined train+val).
        """
        logger = get_logger()

        try:
            import optuna

            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            logger.warning("Optuna not available, using default parameters")
            self._fit_model(X, y, None, None)
            return

        # Internal CV split
        split_idx = int(len(X) * (1 - self.optuna_val_ratio))
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_val = X.iloc[split_idx:]
        y_val = y.iloc[split_idx:]

        logger.debug(
            f"XGBoost Optuna internal split: train={len(X_train)}, "
            f"val={len(X_val)} (ratio={self.optuna_val_ratio})"
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            model = xgb.XGBRegressor(
                **params,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train, verbose=False)

            predictions = model.predict(X_val)
            mape = np.mean(np.abs((y_val - predictions) / (y_val + 1e-8)))
            return mape

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.optuna_trials, show_progress_bar=False)

        best = study.best_params
        self.n_estimators = best["n_estimators"]
        self.max_depth = best["max_depth"]
        self.learning_rate = best["learning_rate"]
        self._optuna_trials_used = len(study.trials)

        # Fit final model on ALL data with best params
        self.model = xgb.XGBRegressor(
            n_estimators=best["n_estimators"],
            max_depth=best["max_depth"],
            learning_rate=best["learning_rate"],
            min_child_weight=best["min_child_weight"],
            subsample=best["subsample"],
            colsample_bytree=best["colsample_bytree"],
            reg_alpha=best["reg_alpha"],
            reg_lambda=best["reg_lambda"],
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X, y, verbose=False)

        logger.info(f"XGBoost Optuna optimization complete: {best}")

    def predict(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Generate forecasts.

        Uses recursive prediction: predict one step, use as lag for next step.

        Args:
            horizon: Number of periods to forecast.
            exog_future: Optional exogenous features for forecast period.

        Returns:
            Forecasted values with proper DatetimeIndex.
        """
        logger = get_logger()

        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before prediction")

        if self._train_data is None:
            raise RuntimeError("No training data available")

        # Get last date from training data
        last_date = self._train_data.index[-1]
        future_index = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS",
        )

        # Start with training data for lag computation
        history = self._train_data.copy()
        predictions = []

        for i in range(horizon):
            # Add future exog features if available
            next_date = future_index[i]
            next_row = pd.DataFrame(index=[next_date])

            if exog_future is not None and next_date in exog_future.index:
                for col in exog_future.columns:
                    next_row[col] = exog_future.loc[next_date, col]

            # Add placeholder for target (will compute lags from history)
            next_row[self._target_col] = np.nan

            # Combine with history
            temp_df = pd.concat([history, next_row])

            # Prepare features for prediction
            temp_features = self._prepare_features(temp_df, is_training=False)

            # Get the last row (our prediction point)
            X_pred = temp_features[self._feature_columns].iloc[[-1]]

            # Fill any NaN values (from insufficient history) with 0
            X_pred = X_pred.fillna(0)

            # Make prediction
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)

            # Update history with prediction
            history.loc[next_date, self._target_col] = pred

        return pd.Series(predictions, index=future_index, name="XGBoost_forecast")

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        if not self._is_fitted or self.model is None:
            return {}

        importance = self.model.feature_importances_
        return dict(zip(self._feature_columns, importance))

    def get_params(self) -> dict:
        """Get model parameters for serialization."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lags": self.lags,
            "rolling_windows": self.rolling_windows,
            "use_optuna": self.use_optuna,
            "optuna_trials": self.optuna_trials,
            "optuna_val_ratio": self.optuna_val_ratio,
            "feature_columns": self._feature_columns,
            "optuna_trials_used": self._optuna_trials_used,
        }

    def load_params(self, params: dict) -> None:
        """Load model parameters."""
        self.n_estimators = params.get("n_estimators", self.n_estimators)
        self.max_depth = params.get("max_depth", self.max_depth)
        self.learning_rate = params.get("learning_rate", self.learning_rate)
        self.lags = params.get("lags", self.lags)
        self.rolling_windows = params.get("rolling_windows", self.rolling_windows)
        self.use_optuna = params.get("use_optuna", self.use_optuna)
        self.optuna_trials = params.get("optuna_trials", self.optuna_trials)
        self.optuna_val_ratio = params.get("optuna_val_ratio", self.optuna_val_ratio)
        self._feature_columns = params.get("feature_columns", [])
        self._optuna_trials_used = params.get("optuna_trials_used", 0)

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
