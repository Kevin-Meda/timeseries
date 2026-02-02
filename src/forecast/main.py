"""CLI entry point for the demand forecasting pipelines."""

import argparse
import sys

from forecast.pipeline import run_pipeline
from forecast.pipeline_univariate import run_univariate_pipeline
from forecast.pipeline_multivariate_single import run_multivariate_single_pipeline
from forecast.pipeline_multivariate_all import run_multivariate_all_pipeline


def main() -> int:
    """Main entry point for the forecasting CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Demand Forecasting Pipeline - Train models and generate forecasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Types:
  default              Original pipeline with all models
  univariate           SARIMA and Holt-Winters only (no features)
  multivariate_single  Prophet, XGBoost, Chronos with feature store features
  multivariate_all     Prophet, XGBoost, Chronos with feature store + cross-product features

Examples:
  forecast                                       Run default pipeline
  forecast --pipeline univariate                 Run univariate pipeline
  forecast --pipeline multivariate_single        Run with feature store features
  forecast --pipeline multivariate_all           Run with cross-product features
  forecast --config-dir ./myconfig               Use custom config directory
  forecast --input data/sales.xlsx               Override input file
  forecast --log-level DEBUG                     Enable debug logging
  forecast --project-name my_project             Organize outputs under project name

Output Directories:
  default:              output/{project}/
  univariate:           output/univariate/{project}/
  multivariate_single:  output/multivariate_single/{project}/
  multivariate_all:     output/multivariate_all/{project}/

Configuration files should be placed in the config directory:
  - data_input.yaml: Input file settings and feature store path
  - preprocessing.yaml: Data cleaning parameters
  - models.yaml: Model selection and tuning
  - pipeline.yaml: Train/val/test splits
  - plots.yaml: Visualization settings
  - output.yaml: Output directories
        """,
    )

    parser.add_argument(
        "--pipeline",
        default="default",
        choices=["default", "univariate", "multivariate_single", "multivariate_all"],
        help="Pipeline type to run (default: default)",
    )

    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Path to configuration directory (default: configs)",
    )

    parser.add_argument(
        "--input",
        dest="input_file",
        help="Override input Excel file path",
    )

    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Override output directory for results",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--project-name",
        dest="project_name",
        default="default",
        help="Project name for organizing outputs (default: default)",
    )

    args = parser.parse_args()

    # Select pipeline function based on type
    pipeline_runners = {
        "default": run_pipeline,
        "univariate": run_univariate_pipeline,
        "multivariate_single": run_multivariate_single_pipeline,
        "multivariate_all": run_multivariate_all_pipeline,
    }

    run_func = pipeline_runners[args.pipeline]

    try:
        results = run_func(
            config_dir=args.config_dir,
            input_override=args.input_file,
            output_dir_override=args.output_dir,
            log_level=args.log_level,
            project_name=args.project_name,
        )

        if "error" in results:
            print(f"Pipeline failed: {results['error']}", file=sys.stderr)
            return 1

        pipeline_type = results.get("pipeline_type", args.pipeline)
        print(f"\n{pipeline_type.replace('_', ' ').title()} pipeline completed successfully!")
        print(f"Categories processed: {results.get('categories_processed', 0)}")

        return 0

    except KeyboardInterrupt:
        print("\nPipeline interrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Pipeline failed with error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
