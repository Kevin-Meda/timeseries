"""Pytest configuration and shared fixtures."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the test data directory."""
    return project_root / "data" / "input"


@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp(prefix="forecast_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="function")
def temp_config_dir(project_root):
    """Create a temporary config directory with test configurations."""
    temp_dir = tempfile.mkdtemp(prefix="forecast_config_")

    # Copy base configs
    src_config = project_root / "configs"
    for config_file in src_config.glob("*.yaml"):
        shutil.copy(config_file, temp_dir)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def feature_store_path(test_data_dir):
    """Create toy feature store and return its path."""
    feature_store_file = test_data_dir / "features.xlsx"

    # Generate date range matching the demand data
    dates = pd.date_range(start="2021-01-01", periods=48, freq="MS")

    np.random.seed(42)

    # Create feature data for each product
    data = {"Date": dates}

    products = ["Product_A", "Product_B", "Product_C"]
    for product in products:
        # Numeric feature: price (with some variation)
        base_price = 100 + np.random.randint(0, 50)
        data[f"{product}_price"] = base_price + np.random.randn(len(dates)) * 5

        # Numeric feature: marketing spend
        data[f"{product}_marketing"] = np.random.uniform(1000, 5000, len(dates))

        # Categorical feature: promo type
        promo_types = ["none", "discount", "bundle", "seasonal"]
        data[f"{product}_promo"] = np.random.choice(promo_types, len(dates))

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)

    # Write to Excel
    df.to_excel(feature_store_file)

    yield str(feature_store_file)

    # Don't delete - keep for manual testing
    # os.remove(feature_store_file)


@pytest.fixture
def univariate_config(temp_config_dir):
    """Create config for univariate testing (no feature store)."""
    import yaml

    # Update data_input.yaml
    data_input_path = Path(temp_config_dir) / "data_input.yaml"
    with open(data_input_path, "r") as f:
        config = yaml.safe_load(f)

    config["feature_store_path"] = None
    config["categorical_features"] = []

    with open(data_input_path, "w") as f:
        yaml.dump(config, f)

    # Update models.yaml for faster testing
    models_path = Path(temp_config_dir) / "models.yaml"
    with open(models_path, "r") as f:
        config = yaml.safe_load(f)

    config["sarima"]["enabled"] = True
    config["sarima"]["optimize_params"] = False
    config["holt_winters"]["enabled"] = True
    config["chronos"]["enabled"] = False  # Skip for faster tests
    config["xgboost"]["enabled"] = False
    config["xgboost"]["optimize_params"] = False
    config["prophet"]["enabled"] = False
    config["prophet"]["optimize_params"] = False

    with open(models_path, "w") as f:
        yaml.dump(config, f)

    return temp_config_dir


@pytest.fixture
def multivariate_config(temp_config_dir, feature_store_path):
    """Create config for multivariate testing (with feature store)."""
    import yaml

    # Update data_input.yaml
    data_input_path = Path(temp_config_dir) / "data_input.yaml"
    with open(data_input_path, "r") as f:
        config = yaml.safe_load(f)

    config["feature_store_path"] = feature_store_path
    config["categorical_features"] = ["promo"]

    with open(data_input_path, "w") as f:
        yaml.dump(config, f)

    # Update models.yaml for multivariate models
    models_path = Path(temp_config_dir) / "models.yaml"
    with open(models_path, "r") as f:
        config = yaml.safe_load(f)

    config["sarima"]["enabled"] = True
    config["sarima"]["optimize_params"] = False
    config["holt_winters"]["enabled"] = True
    config["chronos"]["enabled"] = False
    config["xgboost"]["enabled"] = True
    config["xgboost"]["optimize_params"] = True
    config["xgboost"]["optuna_trials"] = 5  # Minimal for testing
    config["prophet"]["enabled"] = False
    config["prophet"]["optimize_params"] = False

    with open(models_path, "w") as f:
        yaml.dump(config, f)

    return temp_config_dir


# ============================================================================
# 4-Product Test Fixtures
# ============================================================================


@pytest.fixture(scope="function")
def four_product_demand_file(temp_output_dir):
    """Create synthetic demand data for 4 products with different patterns.

    Creates Excel file in the format expected by loader.py:
    - Row 0: dates in format MM.YYYY
    - Column A: product names
    - Columns B onwards: demand values

    Products:
    - Trending: steady upward trend
    - Seasonal: strong seasonal pattern
    - Declining: downward trend
    - StepChange: level shift mid-series
    """
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")

    # Create demand data for each product
    products_data = {}

    # Trending product - steady upward trend
    trend = np.linspace(100, 200, len(dates))
    noise = np.random.randn(len(dates)) * 10
    products_data["Trending"] = (trend + noise).clip(min=1)

    # Seasonal product - strong 12-month seasonality
    base = 150
    seasonal = 50 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
    noise = np.random.randn(len(dates)) * 8
    products_data["Seasonal"] = (base + seasonal + noise).clip(min=1)

    # Declining product - downward trend
    decline = np.linspace(200, 100, len(dates))
    noise = np.random.randn(len(dates)) * 12
    products_data["Declining"] = (decline + noise).clip(min=1)

    # StepChange product - level shift at month 30
    base_level = np.where(np.arange(len(dates)) < 30, 100, 180)
    noise = np.random.randn(len(dates)) * 15
    products_data["StepChange"] = (base_level + noise).clip(min=1)

    # Build DataFrame in loader.py expected format
    # Row 0 = dates, Column A = product names
    date_strings = [d.strftime("%m.%Y") for d in dates]

    # Build data rows: first element is product name, rest are values
    data_rows = []
    # First row is header with dates
    data_rows.append([""] + date_strings)

    # Add product rows
    for product_name, values in products_data.items():
        row = [product_name] + list(values)
        data_rows.append(row)

    df = pd.DataFrame(data_rows)

    # Write to Excel without headers/index
    demand_file = Path(temp_output_dir) / "four_product_demand.xlsx"
    df.to_excel(demand_file, header=False, index=False)

    return str(demand_file)


@pytest.fixture(scope="function")
def four_product_feature_store(temp_output_dir):
    """Create feature store for 4 products with price, marketing, and promo features."""
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=60, freq="MS")

    data = {"Date": dates}

    products = ["Trending", "Seasonal", "Declining", "StepChange"]
    promo_types = ["none", "discount", "bundle", "seasonal"]

    for product in products:
        # Numeric feature: price (with some variation)
        base_price = 50 + np.random.randint(0, 30)
        price_values = base_price + np.random.randn(len(dates)) * 5
        data[f"{product}_price"] = price_values.astype(float)

        # Numeric feature: marketing spend
        marketing_values = np.random.uniform(500, 3000, len(dates))
        data[f"{product}_marketing"] = marketing_values.astype(float)

        # Categorical feature: promo type - use string type explicitly
        promo_values = np.random.choice(promo_types, len(dates))
        data[f"{product}_promo"] = list(promo_values)  # Convert to list for proper string handling

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)

    # Ensure numeric columns are float
    for col in df.columns:
        if "price" in col or "marketing" in col:
            df[col] = df[col].astype(float)

    # Write to Excel
    feature_file = Path(temp_output_dir) / "four_product_features.xlsx"
    df.to_excel(feature_file)

    return str(feature_file)


@pytest.fixture
def four_product_config(temp_config_dir, four_product_demand_file, four_product_feature_store):
    """Create config for 4-product full pipeline test with all models enabled."""
    import yaml

    # Update data_input.yaml
    data_input_path = Path(temp_config_dir) / "data_input.yaml"
    with open(data_input_path, "r") as f:
        config = yaml.safe_load(f) or {}

    config["excel_path"] = four_product_demand_file
    config["feature_store_path"] = four_product_feature_store
    config["categorical_features"] = ["promo"]

    with open(data_input_path, "w") as f:
        yaml.dump(config, f)

    # Update models.yaml - enable all models with optimize_params=true
    models_path = Path(temp_config_dir) / "models.yaml"
    with open(models_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # SARIMA with optimization
    config["sarima"] = {
        "enabled": True,
        "optimize_params": True,
        "optuna_trials": 5,  # Minimal for testing
        "defaults": {"order": [1, 1, 1], "seasonal_order": [1, 1, 1, 12]},
    }

    # Holt-Winters (no optimize_params - uses built-in)
    config["holt_winters"] = {
        "enabled": True,
        "defaults": {"trend": "add", "seasonal": "add", "seasonal_periods": 12},
    }

    # Chronos disabled for faster tests
    config["chronos"] = {
        "enabled": False,
        "defaults": {"model_name": "amazon/chronos-t5-small"},
    }

    # XGBoost disabled by default (can cause issues with feature types in some envs)
    config["xgboost"] = {
        "enabled": False,
        "optimize_params": True,
        "optuna_trials": 5,  # Minimal for testing
        "defaults": {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.1},
    }

    # Prophet disabled by default (can cause issues with feature types)
    config["prophet"] = {
        "enabled": False,
        "optimize_params": True,
        "optuna_trials": 5,  # Minimal for testing
        "defaults": {
            "changepoint_prior_scale": 0.05,
            "seasonality_prior_scale": 10.0,
            "seasonality_mode": "additive",
        },
    }

    with open(models_path, "w") as f:
        yaml.dump(config, f)

    # Update pipeline.yaml for shorter test duration
    pipeline_path = Path(temp_config_dir) / "pipeline.yaml"
    if pipeline_path.exists():
        with open(pipeline_path, "r") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    config["validation_months"] = 6
    config["test_months"] = 6
    config["forecast_horizon"] = 6
    config["ensemble_mape_threshold"] = 0.5

    with open(pipeline_path, "w") as f:
        yaml.dump(config, f)

    return temp_config_dir
