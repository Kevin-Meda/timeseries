"""Feature store loader for external features."""

import pandas as pd
from pathlib import Path
from typing import Any

from forecast.utils import get_logger


class FeatureStoreLoader:
    """Load and join features from an Excel feature store.

    The feature store is expected to have:
    - Date rows (DatetimeIndex)
    - Product-prefixed columns: e.g., Product_A_price, Product_A_promo
    """

    def __init__(self, feature_store_path: str | Path):
        """Initialize feature store loader.

        Args:
            feature_store_path: Path to the feature store Excel file.
        """
        self.feature_store_path = Path(feature_store_path)
        self._features_df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """Load the feature store from Excel.

        Logs file path, row count, and column count.

        Returns:
            DataFrame with DatetimeIndex and all feature columns.

        Raises:
            FileNotFoundError: If the feature store file doesn't exist.
        """
        logger = get_logger()

        if not self.feature_store_path.exists():
            raise FileNotFoundError(
                f"Feature store not found: {self.feature_store_path}"
            )

        # Load Excel with first column as index (dates)
        df = pd.read_excel(
            self.feature_store_path,
            index_col=0,
            parse_dates=True,
        )

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df.index = df.index.to_period("M").to_timestamp()
        df = df.sort_index()

        self._features_df = df

        # Log feature store info
        logger.info(
            f"Feature store loaded: {self.feature_store_path} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )

        return df

    def get_product_features(
        self,
        product_name: str,
        demand_series: pd.Series,
    ) -> pd.DataFrame:
        """Get features for a specific product joined with demand data.

        Logs product name, feature count, and feature names.

        Args:
            product_name: Name of the product (e.g., "Product_A").
            demand_series: Demand time series with DatetimeIndex.

        Returns:
            DataFrame with demand + product-specific features.
        """
        logger = get_logger()

        if self._features_df is None:
            self.load()

        # Find columns for this product
        prefix = f"{product_name}_"
        product_cols = [
            col for col in self._features_df.columns if col.startswith(prefix)
        ]

        if not product_cols:
            # No product-specific features, return just demand
            logger.debug(f"No features found for {product_name}")
            return pd.DataFrame({"demand": demand_series})

        # Extract product features and rename columns (remove prefix)
        product_features = self._features_df[product_cols].copy()
        feature_names = [col.replace(prefix, "") for col in product_features.columns]
        product_features.columns = feature_names

        # Create DataFrame with demand
        demand_df = pd.DataFrame({"demand": demand_series})

        # Join features with demand by date
        result = demand_df.join(product_features, how="left")

        # Log feature info
        logger.info(
            f"Features for {product_name}: {len(feature_names)} features "
            f"({', '.join(feature_names)})"
        )

        return result

    def get_all_product_features(
        self,
        demand_dict: dict[str, pd.Series],
    ) -> dict[str, pd.DataFrame]:
        """Get features for all products.

        Args:
            demand_dict: Dictionary mapping product names to demand series.

        Returns:
            Dictionary mapping product names to DataFrames with demand + features.
        """
        if self._features_df is None:
            self.load()

        result = {}
        for product_name, demand_series in demand_dict.items():
            result[product_name] = self.get_product_features(product_name, demand_series)

        return result

    def extrapolate_features(
        self,
        features_df: pd.DataFrame,
        future_index: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Extrapolate feature values for future periods.

        Uses last known value for each feature column.

        Args:
            features_df: DataFrame with historical features.
            future_index: DatetimeIndex for future periods.

        Returns:
            DataFrame with extrapolated feature values for future dates.
        """
        if features_df.empty or len(future_index) == 0:
            return pd.DataFrame(index=future_index)

        # Get last known values for each feature (excluding 'demand' if present)
        feature_cols = [col for col in features_df.columns if col != "demand"]
        if not feature_cols:
            return pd.DataFrame(index=future_index)

        last_values = features_df[feature_cols].iloc[-1]

        # Create future DataFrame with last values repeated
        future_df = pd.DataFrame(
            {col: [last_values[col]] * len(future_index) for col in feature_cols},
            index=future_index,
        )

        return future_df

    def get_feature_columns(self, product_name: str) -> list[str]:
        """Get list of feature column names for a product.

        Args:
            product_name: Name of the product.

        Returns:
            List of feature column names (without product prefix).
        """
        if self._features_df is None:
            self.load()

        prefix = f"{product_name}_"
        return [
            col.replace(prefix, "")
            for col in self._features_df.columns
            if col.startswith(prefix)
        ]

    def get_categorical_features(
        self,
        product_name: str,
        categorical_prefixes: list[str],
    ) -> list[str]:
        """Get categorical feature names for a product.

        Args:
            product_name: Name of the product.
            categorical_prefixes: List of feature name prefixes that are categorical.

        Returns:
            List of categorical feature column names.
        """
        feature_cols = self.get_feature_columns(product_name)
        categorical = []
        for col in feature_cols:
            for prefix in categorical_prefixes:
                if col.startswith(prefix) or col == prefix:
                    categorical.append(col)
                    break
        return categorical

    def get_numeric_features(
        self,
        product_name: str,
        categorical_prefixes: list[str],
    ) -> list[str]:
        """Get numeric feature names for a product.

        Args:
            product_name: Name of the product.
            categorical_prefixes: List of feature name prefixes that are categorical.

        Returns:
            List of numeric feature column names.
        """
        all_features = self.get_feature_columns(product_name)
        categorical = set(self.get_categorical_features(product_name, categorical_prefixes))
        return [col for col in all_features if col not in categorical]
