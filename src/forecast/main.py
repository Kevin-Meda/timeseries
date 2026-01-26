"""CLI entry point for the demand forecasting pipeline."""

import argparse
import sys

from forecast.pipeline import run_pipeline


def main() -> int:
    """Main entry point for the forecasting CLI.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Demand Forecasting Pipeline - Train models and generate forecasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  forecast                          Run with default configuration
  forecast --config-dir ./myconfig  Use custom config directory
  forecast --input data/sales.xlsx  Override input file
  forecast --log-level DEBUG        Enable debug logging

Configuration files should be placed in the config directory:
  - data_input.yaml: Input file settings
  - preprocessing.yaml: Data cleaning parameters
  - models.yaml: Model selection and tuning
  - pipeline.yaml: Train/val/test splits
  - plots.yaml: Visualization settings
  - output.yaml: Output directories
        """,
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

    args = parser.parse_args()

    try:
        results = run_pipeline(
            config_dir=args.config_dir,
            input_override=args.input_file,
            output_dir_override=args.output_dir,
            log_level=args.log_level,
        )

        if "error" in results:
            print(f"Pipeline failed: {results['error']}", file=sys.stderr)
            return 1

        print(f"\nPipeline completed successfully!")
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
