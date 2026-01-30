"""Permutation importance for time series models."""

from typing import Any, Callable

import numpy as np
import pandas as pd

from forecast.utils import get_logger


class PermutationImportance:
    """Time-blocked permutation importance.

    Respects temporal structure by shuffling within time blocks
    instead of random shuffling across all time points.
    """

    def __init__(
        self,
        model: Any,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
        n_repeats: int = 10,
        block_size: int = 3,
    ):
        """Initialize permutation importance.

        Args:
            model: Fitted model with predict method.
            metric_fn: Function that computes metric (actual, predicted) -> float.
                Lower is better (e.g., MAPE, MSE).
            n_repeats: Number of times to repeat permutation.
            block_size: Size of time blocks for permutation.
        """
        self.model = model
        self.metric_fn = metric_fn
        self.n_repeats = n_repeats
        self.block_size = block_size
        self._importance: dict[str, float] = {}
        self._baseline_score: float | None = None

    def _permute_blocks(self, arr: np.ndarray) -> np.ndarray:
        """Permute array by shuffling blocks.

        Args:
            arr: Array to permute.

        Returns:
            Permuted array with blocks shuffled.
        """
        n = len(arr)
        result = arr.copy()

        # Create block indices
        n_blocks = n // self.block_size
        if n_blocks < 2:
            # If too few blocks, just shuffle randomly
            np.random.shuffle(result)
            return result

        # Shuffle blocks
        block_indices = list(range(n_blocks))
        np.random.shuffle(block_indices)

        permuted = np.zeros_like(arr)
        for i, block_idx in enumerate(block_indices):
            start_src = block_idx * self.block_size
            end_src = min(start_src + self.block_size, n)
            start_dst = i * self.block_size
            end_dst = min(start_dst + self.block_size, n)

            actual_len = min(end_src - start_src, end_dst - start_dst)
            permuted[start_dst : start_dst + actual_len] = arr[start_src : start_src + actual_len]

        # Handle remainder
        remainder_start = n_blocks * self.block_size
        if remainder_start < n:
            permuted[remainder_start:] = arr[remainder_start:]

        return permuted

    def compute(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> dict[str, float]:
        """Compute permutation importance for all features.

        Args:
            X: Feature DataFrame.
            y: Target values.

        Returns:
            Dictionary mapping feature names to importance scores.
        """
        logger = get_logger()

        if isinstance(y, pd.Series):
            y = y.values

        # Baseline score
        baseline_pred = self.model.predict(X)
        self._baseline_score = self.metric_fn(y, baseline_pred)
        logger.debug(f"Baseline score: {self._baseline_score:.4f}")

        importance = {}

        for col in X.columns:
            scores = []

            for _ in range(self.n_repeats):
                # Create copy and permute the column
                X_permuted = X.copy()
                X_permuted[col] = self._permute_blocks(X[col].values)

                # Score with permuted feature
                pred = self.model.predict(X_permuted)
                score = self.metric_fn(y, pred)
                scores.append(score)

            # Importance is the increase in error when feature is permuted
            mean_score = np.mean(scores)
            importance[col] = mean_score - self._baseline_score

        # Normalize by baseline score
        if self._baseline_score != 0:
            importance = {k: v / self._baseline_score for k, v in importance.items()}

        self._importance = importance
        logger.debug(f"Computed permutation importance for {len(importance)} features")

        return importance

    def get_top_features(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N most important features.

        Args:
            n: Number of top features to return.

        Returns:
            List of (feature_name, importance) tuples sorted by importance.
        """
        sorted_features = sorted(
            self._importance.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_features[:n]

    @property
    def importance(self) -> dict[str, float]:
        """Get computed importance dictionary."""
        return self._importance

    @property
    def baseline_score(self) -> float | None:
        """Get baseline score."""
        return self._baseline_score


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    n_repeats: int = 10,
    block_size: int = 3,
) -> dict[str, float]:
    """Convenience function to compute permutation importance.

    Args:
        model: Fitted model with predict method.
        X: Feature DataFrame.
        y: Target values.
        metric_fn: Metric function. Defaults to MAPE.
        n_repeats: Number of permutation repeats.
        block_size: Size of time blocks.

    Returns:
        Dictionary mapping feature names to importance scores.
    """
    if metric_fn is None:
        def mape(actual: np.ndarray, pred: np.ndarray) -> float:
            return np.mean(np.abs((actual - pred) / (actual + 1e-8)))

        metric_fn = mape

    pi = PermutationImportance(
        model=model,
        metric_fn=metric_fn,
        n_repeats=n_repeats,
        block_size=block_size,
    )

    return pi.compute(X, y)
