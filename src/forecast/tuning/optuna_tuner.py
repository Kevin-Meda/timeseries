"""Unified Optuna tuner for all models."""

import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd

from forecast.utils import get_logger

OPTUNA_AVAILABLE = False
try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    pass


def is_optuna_available() -> bool:
    """Check if Optuna is available."""
    return OPTUNA_AVAILABLE


# Search space definitions for each model
SEARCH_SPACES = {
    "SARIMA": {
        "p": {"type": "int", "low": 0, "high": 2},
        "d": {"type": "int", "low": 0, "high": 1},
        "q": {"type": "int", "low": 0, "high": 2},
        "P": {"type": "int", "low": 0, "high": 1},
        "D": {"type": "int", "low": 0, "high": 1},
        "Q": {"type": "int", "low": 0, "high": 1},
    },
    "XGBoost": {
        "n_estimators": {"type": "int", "low": 50, "high": 500},
        "max_depth": {"type": "int", "low": 3, "high": 10},
        "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
        "min_child_weight": {"type": "int", "low": 1, "high": 10},
        "subsample": {"type": "float", "low": 0.6, "high": 1.0},
        "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    },
    "Prophet": {
        "changepoint_prior_scale": {"type": "float", "low": 0.001, "high": 0.5, "log": True},
        "seasonality_prior_scale": {"type": "float", "low": 0.1, "high": 50.0, "log": True},
        "seasonality_mode": {"type": "categorical", "choices": ["additive", "multiplicative"]},
    },
}


class OptunaTuner:
    """Unified Optuna tuner for hyperparameter optimization."""

    def __init__(
        self,
        model_name: str,
        n_trials: int = 50,
        direction: str = "minimize",
    ):
        """Initialize the tuner.

        Args:
            model_name: Name of the model to tune.
            n_trials: Number of Optuna trials.
            direction: "minimize" or "maximize".
        """
        self.model_name = model_name
        self.n_trials = n_trials
        self.direction = direction
        self._study: Any = None
        self._trials_used = 0

    def get_search_space(self) -> dict[str, Any]:
        """Get the search space for this model.

        Returns:
            Dictionary defining the search space.
        """
        return SEARCH_SPACES.get(self.model_name, {})

    def _sample_params(self, trial: Any, search_space: dict[str, Any]) -> dict[str, Any]:
        """Sample parameters from search space.

        Args:
            trial: Optuna trial object.
            search_space: Search space definition.

        Returns:
            Dictionary of sampled parameters.
        """
        params = {}
        for name, config in search_space.items():
            param_type = config["type"]
            if param_type == "int":
                params[name] = trial.suggest_int(name, config["low"], config["high"])
            elif param_type == "float":
                log = config.get("log", False)
                params[name] = trial.suggest_float(
                    name, config["low"], config["high"], log=log
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(name, config["choices"])
        return params

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any]], float],
        search_space: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run Optuna optimization.

        Args:
            objective_fn: Function that takes params dict and returns metric.
            search_space: Optional custom search space. If None, uses default.

        Returns:
            Best parameters found.
        """
        logger = get_logger()

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available, returning empty params")
            return {}

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        if search_space is None:
            search_space = self.get_search_space()

        def objective(trial):
            params = self._sample_params(trial, search_space)
            try:
                return objective_fn(params)
            except Exception:
                return float("inf") if self.direction == "minimize" else float("-inf")

        self._study = optuna.create_study(direction=self.direction)
        self._study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self._trials_used = len(self._study.trials)
        logger.info(
            f"Optuna optimization complete for {self.model_name}: "
            f"{self._trials_used} trials, best params: {self._study.best_params}"
        )

        return self._study.best_params

    @property
    def trials_used(self) -> int:
        """Get number of trials used."""
        return self._trials_used

    @property
    def best_value(self) -> float | None:
        """Get best objective value found."""
        if self._study is not None:
            return self._study.best_value
        return None

    @property
    def best_trial(self) -> Any:
        """Get the best trial object."""
        if self._study is not None:
            return self._study.best_trial
        return None


def create_tuner(model_name: str, n_trials: int = 50) -> OptunaTuner:
    """Factory function to create a tuner for a model.

    Args:
        model_name: Name of the model.
        n_trials: Number of Optuna trials.

    Returns:
        OptunaTuner instance.
    """
    return OptunaTuner(model_name=model_name, n_trials=n_trials)
