"""Data loading and splitting modules."""

from .loader import load_excel
from .splitter import split_series, split_with_features

__all__ = ["load_excel", "split_series", "split_with_features"]
