"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The classes are:
ColumnConfig
    A data structure to hold the name configurations for the analysis.

LabelAccuracy
    Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.

The methods are:
_add_derived_columns
    Adds computed columns for full and partial matches (vectorized).
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
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


@dataclass
class ColumnConfig:
    """A data structure to hold the name configurations for the analysis."""

    model_label_cols: list[str]  # eg: ["model_label_1", "model_label_2"]
    model_score_cols: list[str]  # eg ["model_score_1", "model_score_2"]
    clerical_label_cols: list[str]  # eg ["clerical_label_1", "clerical_label_2"]
    id_col: str = "id"
    filter_unambiguous: bool = False


# pylint: disable=too-few-public-methods
class LabelAccuracy:
    """Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.
    """

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initialises with a dataframe and a configuratoin object, immediately creating derived
        columns for analysis.
        """
        self.config = column_config
        self.id_col = self.config.id_col
        self.model_label_cols = self.config.model_label_cols
        self.model_score_cols = self.config.model_score_cols
        self.clerical_label_cols = self.config.clerical_label_cols

        # Basic validation
        required_cols = [
            self.id_col,
            *self.model_label_cols,
            *self.model_score_cols,
            *self.clerical_label_cols,
        ]

        # If we are filtering on unambiguous, check it is present
        if self.config.filter_unambiguous:
            required_cols.append("Unambiguous")

        if missing_cols := [col for col in required_cols if col not in df.columns]:
            raise ValueError(f"Missing required columns: {missing_cols}")
        if len(self.model_label_cols) != len(self.model_score_cols):
            raise ValueError(
                "Number of model label columns must match number of score columns"
            )

        # If we are filtering on unambiguous, check it is bool, and convert it if not.
        if self.config.filter_unambiguous:
            if df["Unambiguous"].dtype != bool:
                df["Unambiguous"] = (
                    df["Unambiguous"].str.lower().map({"true": True, "false": False})
                )
            df = df[df["Unambiguous"]]

        self.df = df.copy().astype(str, errors="ignore")
        # This one method now calculates all match types
        self._add_derived_columns()

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches (vectorized)."""
        # --- Step 1: Reshape the data from wide to long format ---
        model_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.model_label_cols,
            value_name="model_label",
        ).dropna(subset=["model_label"])

        # Reshape the clerical (ground truth) labels
        clerical_melted = self.df.melt(
            id_vars=[self.id_col],
            value_vars=self.clerical_label_cols,
            value_name="clerical_label",
        ).dropna(subset=["clerical_label"])

        # --- Step 2: Find IDs with at least one FULL match ---
        # Merge the two long dataframes where the ID and the label match exactly
        full_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label"],
            right_on=[self.id_col, "clerical_label"],
        )
        # Get the unique list of IDs that had a match
        full_match_ids = full_matches[self.id_col].unique()

        # --- Step 3: Find IDs with at least one 2-DIGIT match ---
        # Create the 2-digit substring columns before merging
        model_melted["model_label_2_digit"] = model_melted["model_label"].str[:2]
        clerical_melted["clerical_label_2_digit"] = clerical_melted[
            "clerical_label"
        ].str[:2]

        # Merge where the ID and the 2-digit substring match
        partial_matches = pd.merge(
            model_melted,
            clerical_melted,
            left_on=[self.id_col, "model_label_2_digit"],
            right_on=[self.id_col, "clerical_label_2_digit"],
        )
        partial_match_ids = partial_matches[self.id_col].unique()

        # --- Step 4: Map the results back to the original DataFrame ---
        # Create the new boolean columns
        self.df["is_correct"] = self.df[self.id_col].isin(full_match_ids)
        self.df["is_correct_2_digit"] = self.df[self.id_col].isin(partial_match_ids)

        # Also add the max_score column as before
        # Ensure score columns are numeric before finding the max
        for col in self.model_score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        self.df["max_score"] = self.df[self.model_score_cols].max(axis=1)

    def get_accuracy(self, threshold: float = 0.0, match_type: str = "full") -> float:
        """Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold.
            match_type (str): The type of accuracy to calculate.
                              Options: 'full' (default) or '2-digit'.

        Returns:
            float: Accuracy as a percentage.
        """
        if match_type == "2-digit":
            correct_col = "is_correct_2_digit"
        elif match_type == "full":
            correct_col = "is_correct"
        else:
            raise ValueError("match_type must be 'full' or '2-digit'")

        if correct_col not in self.df.columns:
            raise RuntimeError(
                f"Derived column '{correct_col}' not found. Ensure _add_derived_columns ran."
            )

        filtered_df = self.df[self.df["max_score"] >= threshold]
        if len(filtered_df) == 0:
            return 0.0

        return 100 * filtered_df[correct_col].mean()

    def get_coverage(self, threshold: float = 0.0) -> float:
        """Calculate percentage of predictions above the given confidence threshold.

        Args:
            threshold: Minimum confidence score threshold (default: 0.0)

        Returns:
        -------
            float: Coverage as a percentage
        """
        return 100 * (self.df["max_score"] >= threshold).mean()

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

    def plot_threshold_curves(
        self,
        thresholds: Optional[list[float]] = None,
        figsize: tuple[int, int] = (10, 6),
    ) -> None:
        """Plot accuracy and coverage curves against confidence threshold.

        Args:
            thresholds (list[float], optional): List of threshold values to evaluate.
                If None, default thresholds will be used.
            figsize (tuple[int, int], optional): Size of the figure in inches (width, height).
                Defaults to (10, 6).
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

    def plot_confusion_heatmap(
        self,
        human_code_col: str,
        llm_code_col: str,
        top_n: int = 10,
        exclude_patterns: Optional[list[str]] = None,
    ) -> Optional[plt.Axes]:
        """Generates and displays a confusion matrix heatmap for the top N codes.

        Args:
            human_code_col (str): The column name for the ground truth codes.
            llm_code_col (str): The column name for the model's predicted codes.
            top_n (int): The number of most frequent codes to include in the matrix.
            exclude_patterns (list[str]): A list of substrings to filter out from the
                                          human_code_col before analysis (e.g., ['x', '-9']).

        Returns:
            plt.Axes: The matplotlib axes object for further customization.
        """
        # --- Step 1: Create a temporary, smaller DataFrame for efficiency ---
        required_cols = [human_code_col, llm_code_col]
        if any(col not in self.df.columns for col in required_cols):
            raise ValueError(
                "One or both specified columns not found in the DataFrame."
            )

        temp_df = self.df[required_cols].copy()

        # --- Step 2: Clean the data by excluding specified patterns ---
        if exclude_patterns:
            print(f"Initial shape before filtering: {temp_df.shape}")
            for pattern in exclude_patterns:
                temp_df = temp_df[
                    ~temp_df[human_code_col].str.contains(pattern, na=False)
                ]
            print(f"Shape after filtering: {temp_df.shape}")

        # --- Step 3: Find the Most Important Codes to Display ---
        top_human_codes = temp_df[human_code_col].value_counts().nlargest(top_n).index
        top_llm_codes = temp_df[llm_code_col].value_counts().nlargest(top_n).index

        # Filter the DataFrame to only include rows with these top codes
        filtered_df = temp_df[
            temp_df[human_code_col].isin(top_human_codes)
            & temp_df[llm_code_col].isin(top_llm_codes)
        ]

        if filtered_df.empty:
            print(
                "No overlapping data found for the top codes. Cannot generate a matrix."
            )
            return None

        # --- Step 4: Create the Confusion Matrix ---
        confusion_matrix = pd.crosstab(
            filtered_df[human_code_col], filtered_df[llm_code_col]
        )

        # --- Step 5: Visualize as a Heatmap ---
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="YlGnBu")

        plt.title(f"Confusion Matrix: Top {top_n} Human vs. LLM Codes", fontsize=16)
        plt.ylabel(f"Human Coder ({human_code_col})", fontsize=12)
        plt.xlabel(f"LLM Prediction ({llm_code_col})", fontsize=12)
        plt.tight_layout()
        plt.show()

        return heatmap

    @staticmethod
    def save_output(
        metadata: dict, eval_result: dict, save_path: str = "../data/"
    ) -> str:
        """Save evaluation results to files.

        Args:
            metadata: dictionary of metadata parameters
            eval_result: dictionary containing evaluation metrics
            save_path: (str) The folder where results should be saved. Default is "../data/".

        Returns:
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
        with open(metadata_path, "w", encoding="utf-8") as outfile:
            json.dump(metadata, outfile, indent=4)

        # Save evaluation result
        eval_path = os.path.join(folder_name, "evaluation_result.json")
        with open(eval_path, "w", encoding="utf-8") as outfile:
            json.dump(eval_result, outfile, indent=4)

        print(f"Successfully saved all outputs to {folder_name}")
        return folder_name
