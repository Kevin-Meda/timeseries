# Demand Forecasting Platform

A modular demand forecasting platform that reads Excel data, classifies time series, applies preprocessing, trains multiple models (SARIMA, Chronos2, Holt-Winters), evaluates performance, and generates ensemble forecasts with weighted averaging.

## Features

- **Data Loading**: Flexible Excel parsing with configurable date formats
- **Time Series Classification**: Automatically identify forecastable series based on activity
- **Preprocessing**: Handle negatives, NaN, and outliers with neighbor-mean imputation
- **Multiple Models**: SARIMA (with optional Optuna tuning), Chronos2, Holt-Winters
- **Ensemble Forecasting**: Inverse-MAPE weighted averaging
- **Visualization**: Preprocessing, evaluation, and forecast plots
- **Excel Output**: Model results and ensemble forecasts

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
# Install core dependencies
uv sync

# Install with Optuna support
uv sync --extra optuna

# Install with Chronos support (requires PyTorch)
uv sync --extra chronos

# Install all optional dependencies
uv sync --extra all
```

## Quick Start

1. Place your Excel file in `data/input/demand.xlsx`
2. Configure settings in `configs/` directory
3. Run the pipeline:

```bash
uv run forecast
```

## CLI Usage

```bash
# Run with default configuration
uv run forecast

# Use custom config directory
uv run forecast --config-dir ./myconfig

# Override input file
uv run forecast --input data/sales.xlsx

# Override output directory
uv run forecast --output-dir ./results

# Enable debug logging
uv run forecast --log-level DEBUG
```

## Configuration

### data_input.yaml
```yaml
excel_path: "data/input/demand.xlsx"
date_format: "%m.%Y"
product_column: "A"
first_data_column: "B"
```

### preprocessing.yaml
```yaml
classification:
  lookback_months: 24
  min_nonzero_pct: 0.5
data_window_months: 60
outlier_std_threshold: 3.0
```

### models.yaml
```yaml
sarima:
  enabled: true
  use_optuna: false
  optuna_trials: 50
chronos:
  enabled: true
holt_winters:
  enabled: false
```

### pipeline.yaml
```yaml
validation_months: 12
test_months: 12
forecast_horizon: 12
ensemble_mape_threshold: 0.5
```

## Input Format

Excel file structure:
- Row 1: Date headers (e.g., "01.2020", "02.2020", ...)
- Column A: Product/category names
- Column B onwards: Time series values

## Output

### Excel Files
- `output/results/model_results.xlsx`: Per-model metrics (RMSE, MAPE, R²)
- `output/results/ensemble_forecasts.xlsx`: Ensemble predictions with weights

### Plots
- `output/plots/preprocessing/`: Original vs cleaned data
- `output/plots/evaluation/`: Actual vs predicted comparisons
- `output/plots/forecast/`: Historical data + future forecasts

### Model Parameters
- `output/models/`: JSON files with trained model parameters

## Project Structure

```
timeseries/
├── pyproject.toml
├── README.md
├── src/forecast/
│   ├── main.py              # CLI entry point
│   ├── pipeline.py          # Main orchestrator
│   ├── data/                # Data loading/splitting
│   ├── preprocessing/       # Classification and cleaning
│   ├── models/              # Forecasting models
│   ├── evaluation/          # Metrics and ensemble
│   ├── output/              # Excel export
│   ├── plotting/            # Visualizations
│   └── utils/               # Logging
├── configs/                 # YAML configuration files
├── data/input/              # Input Excel files
├── output/
│   ├── results/             # Excel outputs
│   ├── models/              # Model parameters (JSON)
│   └── plots/               # Visualizations
└── logs/                    # Log files
```

## License

MIT
