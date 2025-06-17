"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The class is AlignmentEvaluator

The methods are:
calculate_first_choice_rate
calculate_match_rate_at_n
"""

from typing import Optional
import pandas as pd

# pylint: disable=too-few-public-methods
class AlignmentEvaluator:
    """A class to handle comparison between two sets of classification codes.

    This object loads data and calculates alignment metrics, such as exact
    and partial (n-digit) match rates between any two specified columns.
    """

    def __init__(self, filepath: str):
        """Initialises the evaluator by loading the data.

        Args:
            filepath (str): The path to the CSV file containing the data.
        """
        try:
            self._data = pd.read_csv(filepath, dtype=str)
            print(f"Successfully loaded data with shape: {self._data.shape}")
        except FileNotFoundError:
            print(f"ERROR: The file was not found at {filepath}")
            raise

    def calculate_match_rate(
        self, col1: str, col2: str, n: Optional[int] = None
    ) -> float:
        """Calculates the match rate between two columns, either fully or at n-digits.

        Args:
            col1 (str): The name of the first column to compare.
            col2 (str): The name of the second column to compare.
            n (Optional[int], optional): The number of leading digits to compare.
                If None, a full string comparison is performed. Defaults to None.

        Returns:
            float: The percentage of rows that match, rounded to two decimal places.
        """
        data = self._data
        if col1 not in data.columns or col2 not in data.columns:
            raise ValueError(
                f"One or both columns ('{col1}', '{col2}') not found in data."
            )

        total = len(data)
        if total == 0:
            return 0.0

        # Determine the series to compare based on 'n'
        if n is None:
            # Full match - use the original columns
            series1 = data[col1]
            series2 = data[col2]
        else:
            # Partial match - generate substrings on the fly
            series1 = data[col1].str[:n]
            series2 = data[col2].str[:n]

        matching = (series1 == series2).sum()

        return round(100 * matching / total, 2)


class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of multiple ground truth labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        desc_col: str = "description",
        model_label_cols: list[str] = ["model_label_1", "model_label_2"],
        model_score_cols: list[str] = ["model_score_1", "model_score_2"],
        clerical_label_cols: list[str] = [
            "clerical_label_1",
            "clerical_label_2",
        ],
    ):
        """
        Initialize with a DataFrame containing model predictions and ground truth labels.

        Args:
            df: DataFrame with prediction and ground truth data
            id_col: Name of ID column
            desc_col: Name of description column
            model_label_cols: List of column names containing model predictions
            model_score_cols: List of column names containing confidence scores
            clerical_label_cols: List of column names containing ground truth labels
        """
        self.id_col = id_col
        self.desc_col = desc_col
        self.model_label_cols = model_label_cols
        self.model_score_cols = model_score_cols
        self.clerical_label_cols = clerical_label_cols

        # Verify all required columns exist
        required_cols = (
            [id_col, desc_col]
            + model_label_cols
            + model_score_cols
            + clerical_label_cols
        )
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Verify matching lengths of label and score columns
        if len(model_label_cols) != len(model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

        self.df = df.copy()
        self._add_derived_columns()

    def _add_derived_columns(self):
        """Add computed columns for analysis."""
        # Get highest confidence score among all predictions
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

        # Check if any model prediction matches any clerical label
        def check_matches(row):
            model_labels = [row[col] for col in self.model_label_cols]
            clerical_labels = [row[col] for col in self.clerical_label_cols]
            return any(pred in clerical_labels for pred in model_labels)

        self.df["is_correct"] = self.df.apply(check_matches, axis=1)

    def get_accuracy(self, threshold: float = 0.0) -> float:
        """
        Calculate accuracy for predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns
        -------
            float: Accuracy as a percentage
        """
        filtered_df = self.df[self.df["max_score"] >= threshold]
        if len(filtered_df) == 0:
            return 0.0
        return 100 * filtered_df["is_correct"].mean()            

