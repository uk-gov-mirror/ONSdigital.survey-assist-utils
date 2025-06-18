"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The class is AlignmentEvaluator

The methods are:
calculate_first_choice_rate
calculate_match_rate_at_n
"""

from typing import Optional
from typing import Tuple
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

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
        print('matching', matching, 'total', total)

        return round(100 * matching / total, 2), matching

class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of multiple ground truth labels."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        desc_col: str = "description",
        model_label_cols: list[str] = ["model_label_1", "model_label_2"],
        model_score_cols: list[str] = ["model_score_1", "model_score_2"],
        clerical_label_cols: list[str] = ["clerical_label_1", "clerical_label_2"],
    ):
        """Initializes with a DataFrame, immediately creating derived columns for analysis."""
        self.id_col = id_col
        self.desc_col = desc_col
        self.model_label_cols = model_label_cols
        self.model_score_cols = model_score_cols
        self.clerical_label_cols = clerical_label_cols

        # Basic validation
        required_cols = ([id_col, desc_col] + model_label_cols + model_score_cols + clerical_label_cols)
        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if len(model_label_cols) != len(model_score_cols):
            raise ValueError("Number of model label columns must match number of score columns")

        self.df = df.copy().astype(str, errors='ignore')
        # This one method now efficiently calculates all match types
        self._add_derived_columns_vectorized()

    def _add_derived_columns_vectorized(self):
        """
        Efficiently adds computed columns for full and partial matches using a vectorized approach.
        This is much faster than using df.apply() on large datasets.
        """
        # --- Step 1: Reshape the data from wide to long format ---
        # Reshape the model predictions
        model_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.model_label_cols,
            value_name="model_label"
        ).dropna(subset=["model_label"])

        # Reshape the clerical (ground truth) labels
        clerical_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.clerical_label_cols,
            value_name="clerical_label"
        ).dropna(subset=["clerical_label"])

        # --- Step 2: Find IDs with at least one FULL match ---
        # Merge the two long dataframes where the ID and the label match exactly
        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label"],
            right_on=[self.id_col, "clerical_label"]
        )
        # Get the unique list of IDs that had a match
        full_match_ids = full_matches[self.id_col].unique()

        # --- Step 3: Find IDs with at least one 2-DIGIT match ---
        # Create the 2-digit substring columns before merging
        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted["clerical_label"].str[:2]

        # Merge where the ID and the 2-digit substring match
        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label_2_digit"],
            right_on=[self.id_col, "clerical_label_2_digit"]
        )
        partial_match_ids = partial_matches[self.id_col].unique()

        # --- Step 4: Map the results back to the original DataFrame ---
        # Create the new boolean columns
        self.df["is_correct"] = self.df[self.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.id_col].isin(partial_match_ids)

        # Also add the max_score column as before
        # Ensure score columns are numeric before finding the max
        for col in self.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

    def get_accuracy(self, threshold: float = 0.0, match_type: str = 'full') -> float:
        """
        Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold.
            match_type (str): The type of accuracy to calculate.
                              Options: 'full' (default) or '2-digit'.
        Returns:
            float: Accuracy as a percentage.
        """
        if match_type == '2-digit':
            correct_col = 'is_correct_2_digit'
        elif match_type == 'full':
            correct_col = 'is_correct'
        else:
            raise ValueError("match_type must be 'full' or '2-digit'")

        if correct_col not in self.df.columns:
            raise RuntimeError(f"Derived column '{correct_col}' not found. Ensure _add_derived_columns ran correctly.")

        filtered_df = self.df[self.df["max_score"] >= threshold]
        if len(filtered_df) == 0:
            return 0.0
            
        return 100 * filtered_df[correct_col].mean() 

    def get_coverage(self, threshold: float = 0.0) -> float:
        """
        Calculate percentage of predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns
        -------
            float: Coverage as a percentage
        """
        return 100 * (self.df["max_score"] >= threshold).mean()


    def get_threshold_stats(
        self, thresholds: list[float] = None
    ) -> pd.DataFrame:
        """
        Calculate accuracy and coverage across multiple thresholds.

        Args:
            thresholds: List of threshold values to evaluate (default: None)

        Returns
        -------
            DataFrame with columns: threshold, accuracy, coverage
        """
        if thresholds is None:
            thresholds = np.linspace(0, 1, 21)

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



    def plot_threshold_curves(
        self,
        thresholds: list[float] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> None:
        """
        Plot accuracy and coverage curves against confidence threshold.

        Args:
            thresholds: List of threshold values to evaluate (default: Non
            ))
        """
        stats_df = self.get_threshold_stats(thresholds)

        plt.figure(figsize=figsize)
        plt.plot(
            stats_df["threshold"],
            stats_df["coverage"],
            label="Coverage",
            color="blue",
        )
        plt.plot(
            stats_df["threshold"],
            stats_df["accuracy"],
            label="Accuracy",
            color="orange",
        )

        plt.xlabel("Confidence threshold")
        plt.ylabel("Percentage")
        plt.grid(True)
        plt.legend()
        plt.title("Coverage and Accuracy vs Confidence Threshold")
        plt.tight_layout()
        plt.show()


    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for the classification results.

        Returns
        -------
            Dictionary containing various summary statistics
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


    @staticmethod
    def save_output(
        metadata: dict, eval_result: dict, save_path: str = "../data/"
    ) -> str:
        """Save evaluation results to files.

        Args:
            metadata: Dictionary of metadata parameters
            eval_result: Dictionary containing evaluation metrics
            save_path: (str) The folder where results should be saved. Default is "../data/".

        Returns
        -------
            str: The folder path where results were stored
        """
        if not metadata:
            raise ValueError("Metadata dictionary cannot be empty")

        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

        # Create folder name
        folder_name = os.path.join(
            save_path,
            f"outputs/{formatted_datetime}_{metadata.get('evaluation_type', 'unnamed')}",
        )

        # Create directory safely
        os.makedirs(folder_name, exist_ok=True)

        # Save metadata
        metadata_path = os.path.join(folder_name, "metadata.json")
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile, indent=4)

        # Save evaluation result
        eval_path = os.path.join(folder_name, "evaluation_result.json")
        with open(eval_path, "w") as outfile:
            json.dump(eval_result, outfile, indent=4)

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name