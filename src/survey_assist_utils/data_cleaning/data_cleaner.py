"""data_cleaner.py.

This module defines the `DataCleaner` class, which encapsulates all logic related to
validating, cleaning, and preparing raw evaluation data for downstream analysis or modeling.

Overview:
    The `DataCleaner` class is designed with a single responsibility: to transform a raw
    pandas DataFrame into a cleaned and standardized version based on a provided column
    configuration. It performs input validation, optional filtering of ambiguous records,
    and formatting of label columns, including handling of missing values and code padding.

Usage Example:
    >>> cleaner = DataCleaner(column_config)
    >>> clean_df = cleaner.process(raw_df)

Key Features:
    - Validates the presence and consistency of required columns.
    - Optionally filters out ambiguous records based on a configuration flag.
    - Cleans and standardizes label columns using safe formatting rules.
    - Replaces common missing value formats with `np.nan` to ensure consistency.
    - Preserves the original input DataFrame by operating on a copy.

Dependencies:
    - pandas
    - numpy

See Also:
    - ColumnConfig: Defines the schema and flags used during validation and cleaning.
    - DataCleaner.process: Main entry point for executing the full cleaning pipeline.
"""

# Note - part way through refactoring,
# Linting check disabled temporarily until complete
# pylint: disable=R0801

# This is set to ignore because two linting functions fight with each other.
# One moves it down, the other objects after it was moved.
# "Module level import not at top of file"
# ruff: noqa: E402
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig

_SIC_CODE_PADDING = 5


# pylint: disable=too-few-public-methods
class DataCleaner:
    """Handles the validation, filtering, and cleaning of a raw evaluation DataFrame.

    This class provides a structured pipeline for preparing data for analysis or modeling.
    It uses a configurable column schema to validate required inputs, filter out ambiguous
    records, and standardize label columns.

    Key Methods:
        - `__init__`: Initializes the cleaner with a column configuration.
        - `process`: Main entry point that orchestrates validation, filtering, and cleaning.
        - `_validate_inputs`: Ensures all required columns are present and consistent.
        - `_filter_unambiguous`: Optionally filters rows based on the 'Unambiguous' column.
        - `_clean_dataframe`: Standardizes label columns and handles missing values.

    Attributes:
        _MISSING_VALUE_FORMATS (list[str]): A list of string representations of missing values
        that are replaced with `np.nan` during cleaning.
    """

    _MISSING_VALUE_FORMATS: ClassVar[list[str]] = [
        "",
        " ",
        "nan",
        "None",
        "Null",
        "<NA>",
    ]

    def __init__(self, column_config: ColumnConfig):
        """Initialize the data processing class with a column configuration.

        Args:
            column_config (ColumnConfig): Configuration object specifying column-related
                metadata, including which columns to clean, filter, or validate.
        """
        self.config = column_config

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the full data cleaning and preparation pipeline.

        This method performs the following steps in sequence:
        - Validates that all required inputs and configurations are present.
        - Filters out ambiguous records if configured to do so.
        - Applies cleaning operations to standardize label columns and handle missing values.

        This method serves as the main entry point for preparing the dataset for
        downstream analysis.

        Args:
            df (pd.DataFrame): The input DataFrame to process.

        Returns:
            pd.DataFrame: A cleaned and processed DataFrame ready for analysis.
        """
        working_df = df.copy()
        self._validate_inputs(working_df)
        working_df = self._filter_unambiguous(working_df)
        working_df = self._clean_dataframe(working_df)

        return working_df

    # REFACTOR: This method now accepts a DataFrame to validate against.
    def _validate_inputs(self, df: pd.DataFrame):
        """Validate that all required inputs and configurations are present and consistent.

        This method performs the following checks:
        - Ensures all required columns, as specified in the column configuration,
        are present in the DataFrame.
        - If `filter_unambiguous` is enabled, verifies that the 'Unambiguous' column exists.
        - Confirms that the number of model label columns matches the number of model
        score columns.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If any required columns are missing from the DataFrame.
            ValueError: If the number of model label columns does not match the number of
            model score columns.
        """
        required_cols = [
            self.config.id_col,
            *self.config.model_label_cols,
            *self.config.model_score_cols,
            *self.config.clerical_label_cols,
        ]
        if self.config.filter_unambiguous:
            required_cols.append("Unambiguous")

        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.config.model_label_cols) != len(self.config.model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )
        # Check the data types of the columns:
        for col in self.config.model_score_cols:
            if (
                not pd.api.types.is_numeric_dtype(df[col])
                and not pd.to_numeric(df[col], errors="coerce").notna().all()
            ):
                print(f"Warning: Column '{col}' is not numeric.")

        # Check that label columns are strings or objects that can be treated as strings
        for col in self.config.model_label_cols + self.config.clerical_label_cols:
            if not pd.api.types.is_string_dtype(
                df[col]
            ) and not pd.api.types.is_object_dtype(df[col]):
                print(f"Warning: Label column '{col}' is not of type string or object.")

    # REFACTOR: This method now accepts a DataFrame, filters it, and returns the result.
    def _filter_unambiguous(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter the DataFrame to retain only unambiguous records, if configured to do so.

        If the `filter_unambiguous` flag is enabled in the configuration, this method filters
        the DataFrame based on the 'Unambiguous' column. It handles the following cases:

        - If the 'Unambiguous' column is missing, no filtering is applied.
        - If the column exists but is not of boolean type, it attempts to convert string values
        ('true'/'false') to booleans using a case-insensitive mapping.
        - Rows where 'Unambiguous' is `True` are retained; all others are excluded.

        Args:
            df (pd.DataFrame): The input DataFrame to filter.

        Returns:
            pd.DataFrame: A filtered DataFrame containing only unambiguous records, or the
            original DataFrame if filtering is not applied.
        """
        if self.config.filter_unambiguous:
            if "Unambiguous" not in df.columns:
                return df
            if df["Unambiguous"].dtype != bool:
                df["Unambiguous"] = (
                    df["Unambiguous"].str.lower().map({"true": True, "false": False})
                )
            return df[df["Unambiguous"]]
        return df

    @staticmethod
    # def _safe_zfill(value: Any) -> Any:@staticmethod
    def _safe_zfill(value: Any) -> Any:
        """Safely pad a value with leading zeros to ensure a 5-digit string format.

        This method is typically used to standardize codes (e.g., SIC codes) by converting
        numeric values to zero-padded strings. It handles edge cases gracefully by:

        - Returning the original value if it is `NaN` or in a predefined list of exceptions
        (e.g., "4+", "-9").
        - Attempting to convert the value to a float, then to an integer, and finally to a
        zero-padded string.
        - Returning the original value if conversion fails due to invalid types or formats.

        Args:
            value (Any): The input value to be padded. Can be a number, string, or other type.

        Returns:
            Any: A zero-padded 5-digit string if conversion is successful; otherwise, the
            original value.
        """
        if pd.isna(value) or value in ["4+", "-9"]:
            return value
        try:
            return str(int(float(value))).zfill(_SIC_CODE_PADDING)
        except (ValueError, TypeError):
            return value

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the DataFrame by standardizing label columns and handling missing values.

        This method performs the following operations:
        - Converts all model and clerical label columns to string type for consistency.
        - Replaces predefined missing value formats (e.g., "NA", "-9") with `np.nan`.
        - Applies `_safe_zfill` to each label column to standardize formatting
        (e.g., padding numeric codes and strings like '123x' to '0123x').

        Notes:
            - The columns to be cleaned are determined by combining `model_label_cols` and
            `clerical_label_cols` from the column configuration.
            - `_safe_zfill` ensures consistent formatting of values such as SIC codes.

        Returns:
            pd.DataFrame: A cleaned DataFrame with standardized label columns.
        """
        label_cols = self.config.model_label_cols + self.config.clerical_label_cols
        df[label_cols] = df[label_cols].astype(str)
        df[label_cols] = df[label_cols].replace(self._MISSING_VALUE_FORMATS, np.nan)

        for col in label_cols:
            df[col] = df[col].apply(self._safe_zfill)

        return df
