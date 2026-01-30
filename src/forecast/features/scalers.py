"""Fold-aware scalers to prevent data leakage."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler


class FoldAwareScaler:
    """Scaler that fits only on training data to prevent data leakage.

    Supports RobustScaler and MinMaxScaler from sklearn.
    """

    def __init__(self, scaler_type: str = "robust"):
        """Initialize the scaler.

        Args:
            scaler_type: Type of scaler to use. Options:
                - "robust": RobustScaler (default, uses median and IQR)
                - "minmax": MinMaxScaler (scales to [0, 1])

        Raises:
            ValueError: If scaler_type is not recognized.
        """
        self.scaler_type = scaler_type
        self._scaler: RobustScaler | MinMaxScaler | None = None
        self._numeric_columns: list[str] = []
        self._is_fitted = False

        if scaler_type not in ("robust", "minmax"):
            raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'robust' or 'minmax'.")

    def fit(
        self,
        train_df: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> "FoldAwareScaler":
        """Fit the scaler on training data only.

        Args:
            train_df: Training DataFrame.
            numeric_columns: List of column names to scale.
                If None, auto-detects numeric columns.
            exclude_columns: List of columns to exclude from scaling.

        Returns:
            Self for method chaining.
        """
        exclude = set(exclude_columns or [])

        if numeric_columns is None:
            # Auto-detect numeric columns
            self._numeric_columns = [
                col
                for col in train_df.select_dtypes(include=[np.number]).columns
                if col not in exclude
            ]
        else:
            self._numeric_columns = [
                col
                for col in numeric_columns
                if col in train_df.columns and col not in exclude
            ]

        if not self._numeric_columns:
            self._is_fitted = True
            return self

        # Initialize scaler based on type
        if self.scaler_type == "robust":
            self._scaler = RobustScaler()
        else:
            self._scaler = MinMaxScaler()

        self._scaler.fit(train_df[self._numeric_columns])
        self._is_fitted = True

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform DataFrame using the fitted scaler.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame with scaled numeric columns.

        Raises:
            ValueError: If scaler is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before transform")

        if not self._numeric_columns or self._scaler is None:
            return df.copy()

        result = df.copy()

        # Only transform columns that exist in this DataFrame
        cols_to_transform = [
            col for col in self._numeric_columns if col in result.columns
        ]

        if cols_to_transform:
            if cols_to_transform == self._numeric_columns:
                scaled = self._scaler.transform(result[cols_to_transform])
            else:
                # Handle case where we're transforming subset of columns
                # Create temporary DataFrame with all columns
                temp = pd.DataFrame(
                    index=result.index,
                    columns=self._numeric_columns,
                )
                for col in cols_to_transform:
                    temp[col] = result[col]
                # Fill missing columns with median (neutral for RobustScaler)
                for col in self._numeric_columns:
                    if col not in cols_to_transform:
                        temp[col] = 0

                scaled_full = self._scaler.transform(temp)
                # Extract only the columns we care about
                col_indices = [
                    self._numeric_columns.index(col) for col in cols_to_transform
                ]
                scaled = scaled_full[:, col_indices]

            result[cols_to_transform] = scaled

        return result

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform DataFrame back to original scale.

        Args:
            df: Scaled DataFrame to inverse transform.

        Returns:
            DataFrame with original scale values.

        Raises:
            ValueError: If scaler is not fitted.
        """
        if not self._is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")

        if not self._numeric_columns or self._scaler is None:
            return df.copy()

        result = df.copy()

        cols_to_transform = [
            col for col in self._numeric_columns if col in result.columns
        ]

        if cols_to_transform:
            if cols_to_transform == self._numeric_columns:
                unscaled = self._scaler.inverse_transform(result[cols_to_transform])
            else:
                # Handle subset case
                temp = pd.DataFrame(
                    index=result.index,
                    columns=self._numeric_columns,
                )
                for col in cols_to_transform:
                    temp[col] = result[col]
                for col in self._numeric_columns:
                    if col not in cols_to_transform:
                        temp[col] = 0

                unscaled_full = self._scaler.inverse_transform(temp)
                col_indices = [
                    self._numeric_columns.index(col) for col in cols_to_transform
                ]
                unscaled = unscaled_full[:, col_indices]

            result[cols_to_transform] = unscaled

        return result

    def fit_transform(
        self,
        train_df: pd.DataFrame,
        numeric_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fit the scaler and transform training data.

        Args:
            train_df: Training DataFrame.
            numeric_columns: List of column names to scale.
            exclude_columns: List of columns to exclude from scaling.

        Returns:
            Transformed training DataFrame.
        """
        self.fit(train_df, numeric_columns, exclude_columns)
        return self.transform(train_df)

    @property
    def numeric_columns(self) -> list[str]:
        """Get list of numeric columns being scaled."""
        return self._numeric_columns

    @property
    def is_fitted(self) -> bool:
        """Check if scaler is fitted."""
        return self._is_fitted

    def get_params(self) -> dict:
        """Get scaler parameters for serialization.

        Returns:
            Dictionary with scaler parameters.
        """
        if not self._is_fitted or self._scaler is None:
            return {"scaler_type": self.scaler_type, "columns": []}

        params = {
            "scaler_type": self.scaler_type,
            "columns": self._numeric_columns,
        }

        if self.scaler_type == "robust":
            params["center"] = self._scaler.center_.tolist()
            params["scale"] = self._scaler.scale_.tolist()
        else:
            params["min"] = self._scaler.min_.tolist()
            params["scale"] = self._scaler.scale_.tolist()

        return params
