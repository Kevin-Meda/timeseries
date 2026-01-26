"""Excel data loading module."""

import pandas as pd
from pathlib import Path
from datetime import datetime

from forecast.utils import get_logger


def load_excel(path: str, config: dict) -> dict[str, pd.Series]:
    """Load Excel file and parse time series data.

    Args:
        path: Path to Excel file.
        config: Configuration dict with keys:
            - date_format: Format string for parsing dates (e.g., "%m.%Y")
            - product_column: Column containing product/category names
            - first_data_column: First column containing time series data

    Returns:
        Dictionary mapping category names to pandas Series with DatetimeIndex.
    """
    logger = get_logger()
    logger.info(f"Loading Excel file: {path}")

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")

    # Read all data as strings first, then convert
    df = pd.read_excel(path, header=None, dtype=str)
    logger.debug(f"Loaded DataFrame with shape: {df.shape}")

    date_format = config.get("date_format", "%m.%Y")
    product_col = config.get("product_column", "A")
    first_data_col = config.get("first_data_column", "B")

    product_col_idx = _column_letter_to_index(product_col)
    first_data_col_idx = _column_letter_to_index(first_data_col)

    date_row = df.iloc[0, first_data_col_idx:].values
    dates = _parse_dates(date_row, date_format)
    logger.info(f"Parsed {len(dates)} dates from {dates[0]} to {dates[-1]}")

    result: dict[str, pd.Series] = {}

    for row_idx in range(1, len(df)):
        category_name = str(df.iloc[row_idx, product_col_idx])

        if pd.isna(category_name) or category_name.strip() == "":
            continue

        values = df.iloc[row_idx, first_data_col_idx:].values
        values = pd.to_numeric(values, errors="coerce")

        series = pd.Series(values, index=dates, name=category_name)

        # Keep NaN values for preprocessing to handle
        if len(series.dropna()) > 0:
            result[category_name] = series
            logger.debug(f"Loaded category '{category_name}' with {len(series)} values")

    logger.info(f"Loaded {len(result)} categories from Excel file")
    return result


def _column_letter_to_index(letter: str) -> int:
    """Convert Excel column letter to zero-based index.

    Args:
        letter: Column letter (A, B, C, ..., AA, AB, etc.)

    Returns:
        Zero-based column index.
    """
    result = 0
    for char in letter.upper():
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def _parse_dates(date_values: list, date_format: str) -> pd.DatetimeIndex:
    """Parse date values from Excel row.

    Args:
        date_values: List of date values (strings or datetime objects).
        date_format: Format string for parsing date strings.

    Returns:
        DatetimeIndex of parsed dates.
    """
    from forecast.utils import get_logger
    logger = get_logger()
    parsed_dates = []

    for val in date_values:
        if pd.isna(val):
            continue

        parsed = None

        if isinstance(val, datetime):
            parsed = val
        elif isinstance(val, str):
            # Try the specified format first
            try:
                parsed = datetime.strptime(val.strip(), date_format)
            except ValueError:
                # Try common alternative formats
                for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y/%m/%d"]:
                    try:
                        parsed = datetime.strptime(val.strip(), fmt)
                        break
                    except ValueError:
                        continue
        elif isinstance(val, (int, float)):
            # First check if it looks like M.YYYY or MM.YYYY (e.g., 1.2019 or 12.2019)
            # These are dates that Excel converted to decimals
            str_val = str(val)
            if "." in str_val:
                parts = str_val.split(".")
                if len(parts) == 2:
                    try:
                        month_part = int(parts[0])
                        year_part = int(parts[1])
                        if 1 <= month_part <= 12 and 1900 <= year_part <= 2100:
                            parsed = datetime(year_part, month_part, 1)
                    except (ValueError, TypeError):
                        pass

            # If still not parsed, check if it's an Excel serial date
            if parsed is None and 1 < val < 100000:
                try:
                    parsed = pd.to_datetime(val, unit="D", origin="1899-12-30")
                except (ValueError, TypeError):
                    pass

            # If not parsed yet, try as a compact date format like 12019 -> 01.2019
            if parsed is None:
                str_val = str(int(val)) if isinstance(val, float) and val.is_integer() else str(val)
                # Try MM.YYYY without the dot -> MMYYYY
                if len(str_val) == 6:  # MMYYYY
                    try:
                        parsed = datetime.strptime(str_val, "%m%Y")
                    except ValueError:
                        pass
                elif len(str_val) == 5:  # MYYYY (single digit month)
                    try:
                        parsed = datetime.strptime("0" + str_val, "%m%Y")
                    except ValueError:
                        pass

        if parsed is None:
            try:
                parsed = pd.to_datetime(val)
            except (ValueError, TypeError):
                logger.debug(f"Could not parse date: {val} (type: {type(val).__name__})")
                continue

        if parsed is not None:
            parsed_dates.append(parsed)

    return pd.DatetimeIndex(parsed_dates)
