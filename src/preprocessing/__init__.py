"""
Preprocessing module for PdM data transformation.

This module handles:
- Loading raw PdM CSV files
- Merging telemetry, errors, failures, and maintenance data
- Aggregating to daily time series per machine
- Feature engineering
"""

from .data_merger import DataMerger
