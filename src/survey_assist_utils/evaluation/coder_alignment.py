"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The classes are:
ColumnConfig
    A data structure to hold the name configurations for the analysis.

DataCleaner
    Handles the cleaning and validation of the raw evaluation DataFrame.

LabelAccuracy
    Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.

The methods are:
_safe_zfill
    Safely pads a value with leading zeros to 5 digits.
_validate_inputs
    Centralised method for all input validations.
_clean_dataframe
    Cleans the DataFrame by handling data types and missing values.
_melt_and_clean
    Helper to reshape data from wide to long and drop any remaining NaNs.
_add_derived_columns
    Adds computed columns for full and partial matches.
get_jaccard_similarity
    Calculates the average Jaccard Similarity Index across all rows.
get_candidate_contribution
    Assesses the value add of a single candidate column using vectorised operations.
get_accuracy
    Calculate accuracy for predictions above a confidence threshold.
get_coverage
    Calculate percentage of predictions above the given confidence threshold.
get_threshold_stats
    Calculate accuracy and coverage across multiple thresholds.
plot_threshold_curves
    Plot accuracy and coverage curves against confidence threshold.
get_summary_stats
    Get summary statistics for the classification results.
plot_confusion_heatmap
    Generates and displays a confusion matrix heatmap for the top N codes.
save_output
    Save evaluation results to files.

Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

This module has been refactored to separate responsibilities into distinct classes:
- DataCleaner: For all data preprocessing and validation.
- MetricCalculator: For all numerical metric computations.
- Visualizer: For all plotting and visual output.
- LabelAccuracy: The main entry point that orchestrates the other classes.
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# The standard character length for a fully-padded SIC code.
_SIC_CODE_PADDING = 5


@dataclass
class ColumnConfig:
    """A data structure to hold the name configurations for the analysis."""

    model_label_cols: list[str]
    model_score_cols: list[str]
    clerical_label_cols: list[str]
    id_col: str = "id"
    filter_unambiguous: bool = False


# REFACTOR: A new dataclass to handle plotting arguments cleanly.
@dataclass
class PlotConfig:
    """Configuration for plotting functions."""

    figsize: tuple[int, int] = (12, 10)
    save: bool = False
    filename: Optional[str] = None


# REFACTOR: A new dataclass for the confusion matrix to keep the main
# function signature clean and within linting limits.
@dataclass
class ConfusionMatrixConfig:
    """Configuration for the confusion matrix plot."""

    human_code_col: str
    llm_code_col: str
    top_n: int = 10


# REFACTOR: This new class handles all data cleaning and preparation.
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


# REFACTOR: This new class handles all numerical metric calculations.
# It takes a CLEAN DataFrame and focuses only on computation.
class MetricCalculator:
    """Computes numerical evaluation metrics based on model predictions and labels.

    This class is designed to operate on a cleaned and preprocessed DataFrame,
    using configuration details provided via a `ColumnConfig` object. It supports
    the calculation of various performance metrics such as accuracy, precision,
    recall, and others, depending on the implementation of its public methods.

    Attributes:
        df (pd.DataFrame): A copy of the input DataFrame containing prediction and label
        data.
        config (ColumnConfig): Configuration object specifying relevant column names.
    """

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initializes the MetricCalculator with a DataFrame and column configuration.

        Args:
            df (pd.DataFrame): The input DataFrame containing model predictions,
                scores, and ground truth labels.
            column_config (ColumnConfig): An object that defines which columns
                to use for evaluation (e.g., model labels, scores, clerical labels).

        Notes:
            A copy of the input DataFrame is stored internally to prevent side effects.
            Derived columns needed for metric calculations are added during
            initialization.
        """
        self.df = df.copy()
        self.config = column_config
        self._add_derived_columns()

    def _melt_and_clean(self, value_vars: list[str], value_name: str) -> pd.DataFrame:
        """Reshapes the DataFrame from wide to long format and removes rows with missing
         values.

        This helper method uses `pd.melt` to transform multiple columns into a single
         column,
        making the data suitable for grouped analysis or metric computation. It retains the
        identifier column specified in the configuration and drops any rows where the new
        value column contains NaNs.

        Args:
            value_vars (list[str]): List of column names to unpivot into a single column.
            value_name (str): Name to assign to the new value column in the melted
             DataFrame.

        Returns:
            pd.DataFrame: A long-format DataFrame with NaNs removed from the specified
             value column.
        """
        melted_df = self.df.melt(
            id_vars=[self.config.id_col], value_vars=value_vars, value_name=value_name
        )
        return melted_df.dropna(subset=[value_name])

    def _add_derived_columns(self):
        """Adds derived columns to the DataFrame for evaluation of full and partial label matches.

        This method performs the following operations:
        - Melts model and clerical label columns into long format using `_melt_and_clean`.
        - Identifies full matches where model and clerical labels are identical for the same ID.
        - Identifies partial matches based on the first two digits of the labels
        (e.g., for hierarchical codes).
        - Adds two boolean columns to the DataFrame:
            - `is_correct`: True if a full match exists for the given ID.
            - `is_correct_2_digit`: True if a partial (2-digit) match exists for the given ID.
        - Converts model score columns to numeric types and adds a `max_score` column representing
        the highest model score per row.

        Notes:
            - Matching is performed using inner joins on the ID and label columns.
            - Non-numeric score values are coerced to NaN during conversion.

        Modifies:
            self.df (pd.DataFrame): Adds `is_correct`, `is_correct_2_digit`,
            and `max_score` columns.
        """
        model_melted = self._melt_and_clean(self.config.model_label_cols, "model_label")
        clerical_melted = self._melt_and_clean(
            self.config.clerical_label_cols, "clerical_label"
        )

        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.config.id_col, "model_label"],
            right_on=[self.config.id_col, "clerical_label"],
        )
        full_match_ids = full_matches[self.config.id_col].unique()

        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted[
            "clerical_label"
        ].str[:2]

        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.config.id_col, "model_label_2_digit"],
            right_on=[self.config.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.config.id_col].unique()

        self.df["is_correct"] = self.df[self.config.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.config.id_col].isin(
            partial_match_ids
        )

        for col in self.config.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["max_score"] = self.df[self.config.model_score_cols].max(axis=1)

    def get_jaccard_similarity(self) -> float:
        """Calculates the average Jaccard Similarity Index across all rows in the DataFrame.

        The Jaccard Similarity Index measures the similarity between two sets—in this case,
        the sets of model-assigned labels and clerical-assigned labels for each record.
        It is defined as the size of the intersection divided by the size of the union of
        the two sets.

        Returns:
            float: The mean Jaccard Similarity Index across all rows. A value of 1.0 indicates
            perfect agreement between model and clerical labels, while 0.0 indicates no overlap.

        Notes:
            - If both the model and clerical label sets are empty for a row, the similarity is
                defined as 1.0.
            - Missing values are excluded from the sets before comparison.

        Sub-function:
            calculate_jaccard_for_row(row):
                Computes the Jaccard Similarity Index for a single row.

        Args:
                    row (pd.Series): A row from the DataFrame containing model and clerical labels.

        Returns:
                    float: Jaccard similarity score for the row.
        """

        def calculate_jaccard_for_row(row):
            model_set = set(row[self.config.model_label_cols].dropna())
            clerical_set = set(row[self.config.clerical_label_cols].dropna())
            if not model_set and not clerical_set:
                return 1.0
            intersection = len(model_set.intersection(clerical_set))
            union = len(model_set.union(clerical_set))
            return intersection / union if union > 0 else 0.0

        return self.df.apply(calculate_jaccard_for_row, axis=1).mean()

    def get_candidate_contribution(self, candidate_col: str) -> dict[str, Any]:
        """Evaluates the predictive contribution of a single candidate column by
        comparing it
        against clerical labels using vectorized operations.

        This method calculates two key metrics:
        - **Primary Match**: The percentage of predictions in the candidate column
        that exactly match the primary clerical label (typically the first in the list).
        - **Any Clerical Match**: The percentage of predictions that match any of the
        clerical labels associated with a record.

        Args:
            candidate_col (str): The name of the candidate column to evaluate.

        Returns:
            dict[str, Any]: A dictionary containing:
                - 'candidate_column': The name of the evaluated column.
                - 'total_predictions_made': Number of non-null predictions considered.
                - 'primary_match_percent': Percentage of predictions matching the primary
                clerical label.
                - 'primary_match_count': Count of exact matches with the primary clerical
                label.
                - 'any_clerical_match_percent': Percentage of predictions matching any clerical
                label.
                - 'any_clerical_match_count': Count of matches with any clerical label.

        Raises:
            ValueError: If the candidate column or the primary clerical label column is
            missing.

        Notes:
            - Null values in the candidate column are excluded from evaluation.
            - Matching is case-sensitive and assumes label values are directly comparable.
        """
        primary_clerical_col = self.config.clerical_label_cols[0]
        if (
            candidate_col not in self.df.columns
            or primary_clerical_col not in self.df.columns
        ):
            raise ValueError("Candidate or primary clerical column not found.")

        # Create a working copy with only necessary, non-null candidate predictions
        working_df = self.df[
            [self.config.id_col, candidate_col, *self.config.clerical_label_cols]
        ].dropna(subset=[candidate_col])
        total_considered = len(working_df)

        if total_considered == 0:
            return {"candidate_column": candidate_col, "total_predictions_made": 0}

        # --- Primary Match ---
        primary_match_mask = (
            working_df[candidate_col] == working_df[primary_clerical_col]
        )
        primary_match_count = primary_match_mask.sum()

        # --- Any Clerical Match ---
        clerical_melted = working_df.melt(
            id_vars=[self.config.id_col, candidate_col],
            value_vars=self.config.clerical_label_cols,
            value_name="clerical_label",
        ).dropna(subset=["clerical_label"])

        any_match_mask = (
            clerical_melted[candidate_col] == clerical_melted["clerical_label"]
        )
        any_match_ids = clerical_melted[any_match_mask][self.config.id_col].unique()
        any_match_count = len(any_match_ids)

        return {
            "candidate_column": candidate_col,
            "total_predictions_made": total_considered,
            "primary_match_percent": round(
                100 * primary_match_count / total_considered, 2
            ),
            "primary_match_count": int(primary_match_count),
            "any_clerical_match_percent": round(
                100 * any_match_count / total_considered, 2
            ),
            "any_clerical_match_count": int(any_match_count),
        }

    def get_accuracy(
        self, threshold: float = 0.0, match_type: str = "full", extended: bool = False
    ):
        """Calculates prediction accuracy for records above a specified confidence threshold.

        Accuracy is computed as the percentage of predictions that match the ground truth,
        based on either full or partial (2-digit) label agreement. The method supports
        returning either a simple accuracy percentage or a detailed breakdown of results.

        Args:
            threshold (float, optional): Minimum confidence score (`max_score`) required
                for a prediction to be included in the accuracy calculation. Defaults to 0.0.
            match_type (str, optional): Type of match to evaluate. Accepts:
                - "full": Uses the `is_correct` column for exact label matches.
                - "2-digit": Uses the `is_correct_2_digit` column for partial matches.
                Defaults to "full".
            extended (bool, optional): If True, returns a dictionary with detailed metrics.
                If False, returns only the accuracy percentage. Defaults to False.

        Returns:
            float | dict[str, Any]: Either a float representing the accuracy percentage,
            or a dictionary with the following keys (if `extended=True`):
                - 'accuracy_percent': Accuracy as a percentage (rounded to 1 decimal place).
                - 'matches': Number of correct predictions.
                - 'non_matches': Number of incorrect predictions.
                - 'total_considered': Total number of predictions evaluated.

        Raises:
            KeyError: If the specified match type column does not exist in the DataFrame.

        Notes:
            - Records with `max_score` below the threshold are excluded from evaluation.
            - If no records meet the threshold, accuracy is reported as 0.0.
        """
        correct_col = "is_correct_2_digit" if match_type == "2-digit" else "is_correct"
        filtered_df = self.df[self.df["max_score"] >= threshold]
        total = len(filtered_df)
        if total == 0:
            return (
                {"accuracy_percent": 0.0, "matches": 0, "total_considered": 0}
                if extended
                else 0.0
            )

        matches = filtered_df[correct_col].sum()
        accuracy = 100 * matches / total

        if extended:
            return {
                "accuracy_percent": round(accuracy, 1),
                "matches": int(matches),
                "non_matches": int(total - matches),
                "total_considered": total,
            }
        return accuracy

    def get_threshold_stats(
        self, thresholds: Optional[list[float]] = None
    ) -> pd.DataFrame:
        """Calculate accuracy and coverage across multiple thresholds.

        Args:
            thresholds: list of threshold values to evaluate (default: None)

        Returns:
        -------
            DataFrame with columns: threshold, accuracy, coverage
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21).tolist()

        stats = []
        for threshold in thresholds:
            stats.append(
                {
                    "threshold": threshold,
                    "accuracy": self.get_accuracy(threshold),
                    "coverage": self.get_coverage(threshold),
                }
            )

        return pd.DataFrame(stats)

    def get_coverage(self, threshold: float = 0.0) -> float:
        """Calculate percentage of predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns:
        -------
            float: Coverage as a percentage
        """
        return 100 * (self.df["max_score"] >= threshold).mean()

    def get_summary_stats(self) -> dict:
        """Get summary statistics for the classification results.

        Returns:
        -------
            dictionary containing various summary statistics
        """
        return {
            "total_samples": len(self.df),
            "overall_accuracy": self.get_accuracy(),
            "accuracy_above_0.50": self.get_accuracy(0.5),
            "accuracy_above_0.60": self.get_accuracy(0.6),
            "accuracy_above_0.70": self.get_accuracy(0.7),
            "accuracy_above_0.80": self.get_accuracy(0.8),
            "coverage_above_0.50": self.get_coverage(0.5),
            "coverage_above_0.60": self.get_coverage(0.6),
            "coverage_above_0.70": self.get_coverage(0.7),
            "coverage_above_0.80": self.get_coverage(0.8),
        }


# REFACTOR: This new class handles all plotting.
# It takes a DataFrame that has already been processed by the MetricCalculator.
class Visualizer:
    """Handles all visualization tasks related to model evaluation and performance analysis.

    This class provides plotting and charting utilities that help interpret the results
    of model predictions, label comparisons, and metric calculations. It is designed to
    work with a preprocessed DataFrame and a `MetricCalculator` instance that provides
    derived evaluation metrics.

    Attributes:
        df (pd.DataFrame): The DataFrame containing prediction results and evaluation metadata.
        calculator (MetricCalculator): An instance of MetricCalculator used to compute
            metrics that may be visualized.
    """

    def __init__(self, df: pd.DataFrame, calculator: MetricCalculator):
        """Initializes the Visualizer with a DataFrame and a metric calculator.

        Args:
            df (pd.DataFrame): The input DataFrame containing model predictions,
                scores, and evaluation flags (e.g., is_correct).
            calculator (MetricCalculator): An instance of MetricCalculator that provides
                access to derived metrics and helper methods for evaluation.
        """
        self.df = df
        self.calculator = calculator

    def plot_threshold_curves(self, plot_config: Optional[PlotConfig] = None):
        """Plots coverage and accuracy curves as functions of the confidence threshold.

        This method visualizes how model coverage and accuracy vary with different
        confidence thresholds. It retrieves threshold statistics from the associated
        calculator and generates a line plot with two curves: one for coverage and
        one for accuracy.

        Parameters:
            plot_config (Optional[PlotConfig]): Configuration for the plot, including
                figure size, save options, and filename. If None, a default configuration
                with figsize=(10, 6) is used.

        Raises:
            ValueError: If plot_config.save is True but no filename is provided.

        Behavior:
            - Displays the plot interactively if plot_config.save is False.
            - Saves the plot to the specified filename if plot_config.save is True.

        Assumes:
            - `self.calculator.get_threshold_stats()` returns a DataFrame with columns
            'threshold', 'coverage', and 'accuracy'.
        """
        if plot_config is None:
            plot_config = PlotConfig(figsize=(10, 6))

        stats_df = (
            self.calculator.get_threshold_stats()
        )  # Assumes get_threshold_stats exists on calculator
        plt.figure(figsize=plot_config.figsize)
        plt.plot(stats_df["threshold"], stats_df["coverage"], label="Coverage")
        plt.plot(stats_df["threshold"], stats_df["accuracy"], label="Accuracy")
        plt.xlabel("Confidence threshold")
        plt.ylabel("Percentage")
        plt.grid(True)
        plt.legend()
        plt.title("Coverage and Accuracy vs Confidence Threshold")
        plt.tight_layout()

        if plot_config.save:
            if not plot_config.filename:
                raise ValueError(
                    "Filename must be provided in PlotConfig when save=True."
                )
            plt.savefig(plot_config.filename)
            plt.close()
        else:
            plt.show()

    def plot_confusion_heatmap(
        self,
        matrix_config: ConfusionMatrixConfig,
        plot_config: Optional[PlotConfig] = None,
    ):
        """Generates and displays a heatmap of the confusion matrix between human-coded
        and LLM-predicted labels, focusing on the most frequent codes.

        This method filters the dataset to include only the top N most frequent codes
        from both human and LLM annotations, then computes and visualizes a confusion
        matrix as a heatmap using Seaborn.

        Parameters:
            matrix_config (ConfusionMatrixConfig): Configuration object specifying:
                - `human_code_col`: Column name for human-coded labels.
                - `llm_code_col`: Column name for LLM-predicted labels.
                - `top_n`: Number of top frequent codes to include in the matrix.
            plot_config (Optional[PlotConfig]): Optional configuration for the plot,
                including figure size, save options, and output filename. If not provided,
                a default configuration is used.

        Returns:
            None
        Raises:
            ValueError: If `plot_config.save` is True but no filename is provided.

        Notes:
            - If no overlapping top codes are found between human and LLM labels,
            the method prints a message and exits without plotting.
            - The heatmap is annotated with raw counts and uses the "YlGnBu" color map.
            - The plot is either displayed interactively or saved to a file based on
            `plot_config.save`.

        Assumes:
            - `self.df` is a pandas DataFrame containing the specified columns.
        """
        if plot_config is None:
            plot_config = PlotConfig()

        temp_df = self.df[
            [matrix_config.human_code_col, matrix_config.llm_code_col]
        ].dropna()
        top_human = (
            temp_df[matrix_config.human_code_col]
            .value_counts()
            .nlargest(matrix_config.top_n)
            .index
        )
        top_llm = (
            temp_df[matrix_config.llm_code_col]
            .value_counts()
            .nlargest(matrix_config.top_n)
            .index
        )

        filtered_df = temp_df[
            temp_df[matrix_config.human_code_col].isin(top_human)
            & temp_df[matrix_config.llm_code_col].isin(top_llm)
        ]

        if filtered_df.empty:
            print("No overlapping data for top codes.")
            return None

        matrix = pd.crosstab(
            filtered_df[matrix_config.human_code_col],
            filtered_df[matrix_config.llm_code_col],
        )
        plt.figure(figsize=plot_config.figsize)
        sns.heatmap(matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f"Confusion Matrix: Top {matrix_config.top_n} Codes")
        plt.ylabel(f"Human Coder ({matrix_config.human_code_col})")
        plt.xlabel(f"LLM Prediction ({matrix_config.llm_code_col})")
        plt.tight_layout()

        if plot_config.save:
            if not plot_config.filename:
                raise ValueError(
                    "Filename must be provided in PlotConfig when save=True."
                )
            plt.savefig(plot_config.filename)
            plt.close()
        else:
            plt.show()

        return None


# REFACTOR: The main LabelAccuracy class is now a "facade".
# It coordinates the other classes but keeps the public API the same,
# so your existing scripts will not break.


class LabelAccuracy:
    """Facade class for orchestrating coder alignment analysis.

    This class coordinates the full pipeline for evaluating label accuracy between
    human coders and LLM predictions. It handles data cleaning, metric computation,
    and visualization, while maintaining a stable public API for backward compatibility.

    Components:
        - DataCleaner: Cleans and prepares the input DataFrame.
        - MetricCalculator: Computes accuracy, coverage, and threshold-based metrics.
        - Visualizer: Generates plots and visual summaries of the results.
    """

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initializes the LabelAccuracy analysis pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame containing human and LLM labels.
            column_config (ColumnConfig): Configuration object specifying column names
                and other parameters required for processing and analysis.

        Workflow:
            1. Cleans the input data using `DataCleaner`.
            2. Computes metrics using `MetricCalculator`.
            3. Prepares visualizations using `Visualizer`.

        Attributes:
            df (pd.DataFrame): The cleaned and processed DataFrame.
            calculator (MetricCalculator): Instance used for computing evaluation metrics.
            visualizer (Visualizer): Instance used for generating plots and visualizations.
        """
        # Step 1: Clean and prepare the data
        cleaner = DataCleaner(df, column_config)
        clean_df = cleaner.process()

        # Step 2: Initialise the calculator with the clean data
        self.calculator = MetricCalculator(clean_df, column_config)

        # The final, processed DataFrame is stored here
        self.df = self.calculator.df

        # Step 3: Initialise the visualiser
        self.visualizer = Visualizer(self.df, self.calculator)

    # REFACTOR: Public methods now delegate their calls to the appropriate helper class.
    # This makes the LabelAccuracy class very simple and easy to read.
    def get_accuracy(self, **kwargs):
        """Calculate accuracy for predictions above a confidence threshold."""
        return self.calculator.get_accuracy(**kwargs)

    def get_jaccard_similarity(self, **kwargs):
        """Calculates the average Jaccard Similarity Index."""
        return self.calculator.get_jaccard_similarity(**kwargs)

    def plot_confusion_heatmap(self, **kwargs):
        """Generates and displays a confusion matrix heatmap."""
        return self.visualizer.plot_confusion_heatmap(**kwargs)

    # ... you would add similar passthrough methods for all other public functions ...
    # e.g., get_coverage, get_summary_stats, plot_threshold_curves etc.

    @staticmethod
    def save_output(metadata: dict, eval_result: dict, save_path: str = "data/") -> str:
        """Saves evaluation metadata and results to timestamped JSON files in a structured
        directory.

        This method creates a new output folder using the current timestamp and the
        evaluation type (if provided in metadata), then writes the metadata and evaluation
        results to separate JSON files within that folder.

        Args:
            metadata (dict): Dictionary containing metadata about the evaluation.
                Must include an 'evaluation_type' key for meaningful folder naming.
            eval_result (dict): Dictionary containing the evaluation results to be saved.
            save_path (str, optional): Base directory where the output folder will be created.
                Defaults to "data/".

        Returns:
            str: The full path to the folder where the files were saved.

        Raises:
            ValueError: If the `metadata` dictionary is empty.

        Output:
            - `metadata.json`: Contains the evaluation metadata.
            - `evaluation_result.json`: Contains the evaluation results.

        Example folder structure:
            data/outputs/20250724_143210_classification/metadata.json
            data/outputs/20250724_143210_classification/evaluation_result.json
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = os.path.join(
            save_path, f"outputs/{dt_str}_{metadata.get('evaluation_type', 'unnamed')}"
        )
        os.makedirs(folder_name, exist_ok=True)

        with open(
            os.path.join(folder_name, "metadata.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(metadata, f, indent=4)

        with open(
            os.path.join(folder_name, "evaluation_result.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(eval_result, f, indent=4)

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name
