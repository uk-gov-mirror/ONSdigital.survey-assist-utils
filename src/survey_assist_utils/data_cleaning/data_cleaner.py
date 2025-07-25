"""data_cleaner.py.

This module defines the DataCleaner class, which encapsulates all logic related to
cleaning and preparing raw evaluation data for downstream analysis or modeling.

The class is designed with a single responsibility: to take a raw pandas DataFrame
and return a cleaned version based on a provided column configuration. It performs
input validation, optional filtering of ambiguous records, and standardization of
label columns including handling of missing values and formatting of codes.

Usage:
    cleaner = DataCleaner(raw_df, column_config)
    clean_df = cleaner.process()

Features:
- Validates required columns and configuration consistency.
- Optionally filters out ambiguous records based on a flag.
- Cleans and standardizes label columns using safe formatting rules.
- Handles missing value formats and preserves data integrity.

Note:
    This module disables pylint's 'too-few-public-methods' warning as the class is
    intentionally minimal and focused on a single processing method.
"""

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from survey_assist_utils.configs.column_config import (
    ColumnConfig,
)

_SIC_CODE_PADDING = 5


# Its only responsibility is to take a raw DataFrame and return a clean one.
# pylint: disable=too-few-public-methods
class DataCleaner:
    """Handles the cleaning and validation of the raw evaluation DataFrame."""

    _MISSING_VALUE_FORMATS: ClassVar[list[str]] = [
        "",
        " ",
        "nan",
        "None",
        "Null",
        "<NA>",
    ]

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initializes the data processing class with a DataFrame and column configuration.

        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned and processed.
            column_config (ColumnConfig): Configuration object specifying column-related
                metadata such as which columns to clean, filter, or validate.

        Notes:
            A copy of the input DataFrame is stored internally to avoid modifying the original
            data.
        """
        self.df = df.copy()
        self.config = column_config

    def process(self) -> pd.DataFrame:
        """Executes the full data cleaning and preparation pipeline.

        This method sequentially performs input validation, filters out ambiguous data,
        and applies cleaning operations to the internal DataFrame. It is intended to be
        the main entry point for preparing the dataset for downstream tasks.

        Returns:
            pd.DataFrame: The cleaned and processed DataFrame ready for analysis or modeling.
        """
        self._validate_inputs()
        self._filter_unambiguous()
        self._clean_dataframe()
        return self.df

    def _validate_inputs(self):
        """Validates that all required inputs and configurations are present and consistent.

        This method performs the following checks:
        - Ensures that all required columns, as specified in the column configuration,
        are present in the DataFrame.
        - If `filter_unambiguous` is enabled in the configuration, checks for the presence
        of the 'Unambiguous' column.
        - Verifies that the number of model label columns matches the number of model score columns.

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

        if missing_cols := [col for col in required_cols if col not in self.df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")

        if len(self.config.model_label_cols) != len(self.config.model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

    def _filter_unambiguous(self):
        """Filters the DataFrame to retain only unambiguous records, if configured to do so.

        This method checks whether the `filter_unambiguous` flag is enabled in the column
        configuration. If so, it attempts to filter the DataFrame based on the 'Unambiguous'
        column. If the column exists and is not already of boolean type, it attempts to
        convert string values ('true'/'false') to boolean.

        Notes:
            - If the 'Unambiguous' column is missing, the method exits without filtering.
            - Non-boolean 'Unambiguous' values are converted using a case-insensitive mapping.

        Modifies:
            self.df (pd.DataFrame): Filters rows in-place to include only those where
            'Unambiguous' is True.
        """
        if self.config.filter_unambiguous:
            if "Unambiguous" not in self.df.columns:
                return
            if self.df["Unambiguous"].dtype != bool:
                self.df["Unambiguous"] = (
                    self.df["Unambiguous"]
                    .str.lower()
                    .map({"true": True, "false": False})
                )
            self.df = self.df[self.df["Unambiguous"]]

    @staticmethod
    def _safe_zfill(value: Any) -> Any:
        """Safely pads a numeric value with leading zeros to ensure a 5-digit string format.

        This method is typically used for standardizing codes (e.g., SIC codes) by converting
        numeric values to zero-padded strings. It handles edge cases gracefully by:
        - Returning the original value if it is NaN or in a predefined list of exceptions
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

    def _clean_dataframe(self):
        """Cleans the DataFrame by standardizing label columns and handling missing values.

        This method performs the following operations:
        - Converts all model and clerical label columns to string type to ensure consistency.
        - Replaces predefined missing value formats (e.g., placeholders like "NA", "-9") with
         `np.nan`.
        - Applies `_safe_zfill` to each label column to standardize formatting (e.g., padding
        numeric codes).

        Notes:
            - The columns to be cleaned are determined by combining `model_label_cols` and
            `clerical_label_cols`
            from the column configuration.
            - `_safe_zfill` is used to ensure consistent formatting of values such as SIC
            codes.

        Modifies:
            self.df (pd.DataFrame): Updates label columns in-place with cleaned and
            standardized values.
        """
        label_cols = self.config.model_label_cols + self.config.clerical_label_cols
        self.df[label_cols] = self.df[label_cols].astype(str)
        self.df[label_cols] = self.df[label_cols].replace(
            self._MISSING_VALUE_FORMATS, np.nan
        )

        for col in label_cols:
            self.df[col] = self.df[col].apply(self._safe_zfill)
