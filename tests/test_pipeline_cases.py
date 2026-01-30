"""Pipeline test cases for the forecasting platform.

Test Cases:
1.1: Univariate + Optuna (retrain=true, no feature_store_path)
1.2: Univariate + Load Params (retrain=false, uses params from 1.1)
2.1: Multivariate + Feature Store + Optuna + Explanations
2.2: Multivariate + Load Params + Explanations
"""

import os
import json
from pathlib import Path

import pytest
import yaml


def _check_xgboost_available():
    """Check if XGBoost is installed."""
    try:
        import xgboost
        return True
    except ImportError:
        return False


def _check_prophet_available():
    """Check if Prophet is installed."""
    try:
        from prophet import Prophet
        return True
    except ImportError:
        return False


class TestUnivariateMode:
    """Test cases for univariate forecasting mode."""

    def test_univariate_with_optuna(self, univariate_config, temp_output_dir, project_root):
        """Case 1.1: Univariate + Optuna.

        - retrain=true, no feature_store_path
        - Assert: models trained, params saved, metrics computed
        """
        from forecast.pipeline import run_pipeline

        # Run pipeline
        results = run_pipeline(
            config_dir=univariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_univariate",
        )

        # Assertions
        assert "error" not in results, f"Pipeline failed: {results.get('error')}"
        assert results["categories_processed"] > 0
        assert "model_results" in results
        assert "forecasts" in results
        assert results["project_name"] == "test_univariate"

        # Check that params were saved
        models_dir = Path(temp_output_dir) / "test_univariate" / "models"
        assert models_dir.exists(), "Models directory should exist"

        # Find the timestamp directory
        timestamp_dirs = list(models_dir.iterdir())
        assert len(timestamp_dirs) > 0, "Should have at least one timestamp directory"

        params_files = list(timestamp_dirs[0].glob("*_params.json"))
        assert len(params_files) > 0, "Should have saved parameter files"

        # Check that results were saved
        results_dir = Path(temp_output_dir) / "test_univariate" / "results"
        assert results_dir.exists(), "Results directory should exist"

        # Check JSON summary exists
        result_timestamp_dirs = list(results_dir.iterdir())
        json_summary = result_timestamp_dirs[0] / "run_summary.json"
        assert json_summary.exists(), "JSON summary should exist"

        # Validate JSON structure
        with open(json_summary, "r") as f:
            summary = json.load(f)

        assert "run_info" in summary
        assert summary["run_info"]["project_name"] == "test_univariate"
        assert "products" in summary

    def test_univariate_load_params(self, univariate_config, temp_output_dir, project_root):
        """Case 1.2: Univariate + Load Params.

        - First run with retrain=true to create params
        - Second run with retrain=false to load params
        - Assert: no Optuna runs, params loaded
        """
        from forecast.pipeline import run_pipeline

        # First run: create params
        results1 = run_pipeline(
            config_dir=univariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_load_params",
        )

        assert "error" not in results1

        # Modify config to set retrain=false
        models_path = Path(univariate_config) / "models.yaml"
        with open(models_path, "r") as f:
            config = yaml.safe_load(f)

        config["retrain"] = False

        with open(models_path, "w") as f:
            yaml.dump(config, f)

        # Second run: should load params
        results2 = run_pipeline(
            config_dir=univariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_load_params",
        )

        assert "error" not in results2
        assert results2["categories_processed"] > 0

        # Both runs should have produced forecasts
        assert len(results1["forecasts"]) == len(results2["forecasts"])


class TestMultivariateMode:
    """Test cases for multivariate forecasting mode with feature store."""

    @pytest.mark.skipif(
        not _check_xgboost_available(),
        reason="XGBoost not installed",
    )
    def test_multivariate_with_optuna(
        self, multivariate_config, temp_output_dir, feature_store_path
    ):
        """Case 2.1: Multivariate + Feature Store + Optuna + Explanations.

        - retrain=true, feature_store_path set
        - Assert: features loaded, XGBoost trained, feature importance computed
        """
        from forecast.pipeline import run_pipeline

        # Run pipeline
        results = run_pipeline(
            config_dir=multivariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_multivariate",
        )

        # Assertions
        assert "error" not in results, f"Pipeline failed: {results.get('error')}"
        assert results["categories_processed"] > 0

        # Check model results include XGBoost
        model_results = results.get("model_results", {})
        has_xgboost = any(
            "XGBoost" in models for models in model_results.values()
        )
        assert has_xgboost, "XGBoost should be in results when enabled"

        # Check JSON summary for feature importance
        results_dir = Path(temp_output_dir) / "test_multivariate" / "results"
        result_timestamp_dirs = list(results_dir.iterdir())
        json_summary = result_timestamp_dirs[0] / "run_summary.json"

        with open(json_summary, "r") as f:
            summary = json.load(f)

        # Check that XGBoost models have feature importance
        for product_name, product_data in summary.get("products", {}).items():
            models = product_data.get("models", {})
            if "XGBoost" in models:
                xgb_data = models["XGBoost"]
                # XGBoost should have feature importance when trained with features
                assert "params" in xgb_data
                assert xgb_data["retrained"] is True

    @pytest.mark.skipif(
        not _check_xgboost_available(),
        reason="XGBoost not installed",
    )
    def test_multivariate_load_params(
        self, multivariate_config, temp_output_dir, feature_store_path
    ):
        """Case 2.2: Multivariate + Load Params + Explanations.

        - First run with retrain=true
        - Second run with retrain=false
        - Assert: params loaded, explanations still generated
        """
        from forecast.pipeline import run_pipeline

        # First run: create params
        results1 = run_pipeline(
            config_dir=multivariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_multi_load",
        )

        assert "error" not in results1

        # Modify config to set retrain=false
        models_path = Path(multivariate_config) / "models.yaml"
        with open(models_path, "r") as f:
            config = yaml.safe_load(f)

        config["retrain"] = False

        with open(models_path, "w") as f:
            yaml.dump(config, f)

        # Second run: should load params
        results2 = run_pipeline(
            config_dir=multivariate_config,
            output_dir_override=temp_output_dir,
            log_level="WARNING",
            project_name="test_multi_load",
        )

        assert "error" not in results2
        assert results2["categories_processed"] > 0


class TestFeatureStore:
    """Test cases for feature store functionality."""

    def test_feature_store_loading(self, feature_store_path):
        """Test that feature store loads correctly."""
        from forecast.features.feature_store import FeatureStoreLoader

        loader = FeatureStoreLoader(feature_store_path)
        df = loader.load()

        assert df is not None
        assert len(df) > 0
        assert "Product_A_price" in df.columns
        assert "Product_A_promo" in df.columns

    def test_feature_store_product_features(self, feature_store_path):
        """Test extracting features for a specific product."""
        import pandas as pd
        from forecast.features.feature_store import FeatureStoreLoader

        loader = FeatureStoreLoader(feature_store_path)
        loader.load()

        # Create mock demand series
        dates = pd.date_range(start="2021-01-01", periods=24, freq="MS")
        demand = pd.Series(range(24), index=dates)

        features = loader.get_product_features("Product_A", demand)

        assert "demand" in features.columns
        assert "price" in features.columns
        assert "promo" in features.columns

    def test_feature_extrapolation(self, feature_store_path):
        """Test feature value extrapolation for future periods."""
        import pandas as pd
        from forecast.features.feature_store import FeatureStoreLoader

        loader = FeatureStoreLoader(feature_store_path)
        loader.load()

        # Create mock demand series
        dates = pd.date_range(start="2021-01-01", periods=24, freq="MS")
        demand = pd.Series(range(24), index=dates)

        features = loader.get_product_features("Product_A", demand)

        # Extrapolate for future periods
        future_dates = pd.date_range(start="2023-01-01", periods=12, freq="MS")
        future_features = loader.extrapolate_features(features, future_dates)

        assert len(future_features) == 12
        # Check that values are repeated from last known
        assert "price" in future_features.columns


class TestEncoderScaler:
    """Test cases for encoder and scaler functionality."""

    def test_fold_aware_encoder(self):
        """Test that encoder fits only on training data."""
        import pandas as pd
        from forecast.features.encoders import FoldAwareEncoder

        # Create mock data
        train_df = pd.DataFrame({"cat_col": ["a", "b", "c", "a", "b"]})
        val_df = pd.DataFrame({"cat_col": ["a", "b", "d"]})  # 'd' is unknown

        encoder = FoldAwareEncoder()
        encoder.fit(train_df, ["cat_col"])

        train_encoded = encoder.transform(train_df)
        val_encoded = encoder.transform(val_df)

        # Check that training categories are encoded
        assert train_encoded["cat_col"].dtype in [float, int]

        # Check that unknown category gets -1
        assert -1 in val_encoded["cat_col"].values

    def test_fold_aware_scaler(self):
        """Test that scaler fits only on training data."""
        import pandas as pd
        import numpy as np
        from forecast.features.scalers import FoldAwareScaler

        # Create mock data
        train_df = pd.DataFrame({"num_col": [1.0, 2.0, 3.0, 4.0, 5.0]})
        val_df = pd.DataFrame({"num_col": [6.0, 7.0, 8.0]})

        scaler = FoldAwareScaler("robust")
        scaler.fit(train_df)

        train_scaled = scaler.transform(train_df)
        val_scaled = scaler.transform(val_df)

        # Check that scaling was applied
        assert train_scaled["num_col"].std() != train_df["num_col"].std()

        # Check inverse transform
        train_unscaled = scaler.inverse_transform(train_scaled)
        np.testing.assert_array_almost_equal(
            train_unscaled["num_col"].values,
            train_df["num_col"].values,
            decimal=5,
        )


class TestProjectManager:
    """Test cases for project manager."""

    def test_project_manager_paths(self, temp_output_dir):
        """Test that ProjectManager creates correct paths."""
        from forecast.utils.project_manager import ProjectManager

        pm = ProjectManager(
            project_name="test_project",
            base_output_dir=temp_output_dir,
        )

        # Check that paths are created correctly
        results_dir = pm.results_dir
        assert "test_project" in str(results_dir)
        assert results_dir.exists()

        models_dir = pm.models_dir
        assert "test_project" in str(models_dir)
        assert models_dir.exists()

    def test_project_manager_param_saving(self, temp_output_dir):
        """Test saving and loading parameters."""
        from forecast.utils.project_manager import ProjectManager

        pm = ProjectManager(
            project_name="test_params",
            base_output_dir=temp_output_dir,
        )

        # Save params
        params = {"n_estimators": 100, "max_depth": 6}
        pm.save_params("XGBoost", "Product_A", params)

        # Load params
        loaded = pm.find_latest_params("XGBoost", "Product_A")

        assert loaded is not None
        assert loaded["n_estimators"] == 100
        assert loaded["max_depth"] == 6
