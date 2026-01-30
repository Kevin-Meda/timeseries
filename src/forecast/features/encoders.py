"""Fold-aware encoders to prevent data leakage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


class FoldAwareEncoder:
    """Encoder that fits only on training data to prevent data leakage.

    Wraps sklearn's OrdinalEncoder with proper fit/transform semantics
    for time series cross-validation.
    """

    def __init__(self, handle_unknown: str = "use_encoded_value", unknown_value: int = -1):
        """Initialize the encoder.

        Args:
            handle_unknown: How to handle unknown categories during transform.
                "use_encoded_value" to map to unknown_value.
            unknown_value: Value to use for unknown categories. Default -1.
        """
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self._encoder: OrdinalEncoder | None = None
        self._categorical_columns: list[str] = []
        self._is_fitted = False

    def fit(
        self,
        train_df: pd.DataFrame,
        categorical_columns: list[str] | None = None,
    ) -> "FoldAwareEncoder":
        """Fit the encoder on training data only.

        Args:
            train_df: Training DataFrame.
            categorical_columns: List of column names to encode.
                If None, auto-detects object/category dtype columns.

        Returns:
            Self for method chaining.
        """
        if categorical_columns is None:
            # Auto-detect categorical columns
            self._categorical_columns = list(
                train_df.select_dtypes(include=["object", "category"]).columns
            )
        else:
            self._categorical_columns = [
                col for col in categorical_columns if col in train_df.columns
            ]

        if not self._categorical_columns:
            self._is_fitted = True
            return self

        # Initialize and fit encoder
        self._encoder = OrdinalEncoder(
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value,
        )
        self._encoder.fit(train_df[self._categorical_columns])
        self._is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using the fitted encoder.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with encoded categorical columns.

        Raises:
            ValueError: If encoder is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Encoder must be fitted before transform")

        if not self._categorical_columns or self._encoder is None:
            return df.copy()

        result = df.copy()

        # Only transform columns that exist in this DataFrame
        cols_to_transform = [
            col for col in self._categorical_columns if col in result.columns
        ]

        if cols_to_transform:
            # Need to handle case where we're transforming subset of columns
            if cols_to_transform == self._categorical_columns:
                encoded = self._encoder.transform(result[cols_to_transform])
            else:
                # Create temporary DataFrame with all columns (fill missing with placeholder)
                temp = pd.DataFrame(
                    index=result.index,
                    columns=self._categorical_columns,
                )
                for col in cols_to_transform:
                    temp[col] = result[col]
                # Fill missing columns with first known category
                for i, col in enumerate(self._categorical_columns):
                    if col not in cols_to_transform:
                        temp[col] = self._encoder.categories_[i][0]
                encoded_full = self._encoder.transform(temp)
                # Extract only the columns we care about
                col_indices = [
                    self._categorical_columns.index(col) for col in cols_to_transform
                ]
                encoded = encoded_full[:, col_indices]

            result[cols_to_transform] = encoded

        return result

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        categorical_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit the encoder and transform training data.

        Args:
            train_df: Training DataFrame.
            categorical_columns: List of column names to encode.

        Returns:
            Transformed training DataFrame.
        """
        self.fit(train_df, categorical_columns)
        return self.transform(train_df)

    @property
    def categorical_columns(self) -> list[str]:
        """Get list of categorical columns being encoded."""
        return self._categorical_columns

    @property
    def is_fitted(self) -> bool:
        """Check if encoder is fitted."""
        return self._is_fitted

    def get_categories(self) -> dict[str, list]:
        """Get categories for each encoded column.

        Returns:
            Dictionary mapping column names to their categories.
        """
        if not self._is_fitted or self._encoder is None:
            return {}

        return {
            col: list(cats)
            for col, cats in zip(self._categorical_columns, self._encoder.categories_)
        }
