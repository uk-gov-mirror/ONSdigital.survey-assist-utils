"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
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
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# REFACTOR: Import the moved classes from their new locations.
from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.data_cleaning.data_cleaner import DataCleaner


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


# REFACTOR: The DataCleaner class has been moved to its own file in
# survey_assist_utils/data_cleaning/data_cleaner.py


# REFACTOR: This new class handles all numerical metric calculations.
# It takes a CLEAN DataFrame and focuses only on computation.
class MetricCalculator:
    """Calculates all numerical evaluation metrics."""

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initialises the MetricCalculator.

        Args:
            df (pd.DataFrame): A clean DataFrame, post-DataCleaner.
            column_config (ColumnConfig): The configuration for the analysis.
        """
        self.df = df.copy()
        self.config = column_config
        self._add_derived_columns()

    def _melt_and_clean(self, value_vars: list[str], value_name: str) -> pd.DataFrame:
        """Helper to reshape data from wide to long and drop NaNs."""
        melted_df = self.df.melt(
            id_vars=[self.config.id_col], value_vars=value_vars, value_name=value_name
        )
        return melted_df.dropna(subset=[value_name])

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches."""
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
        """Calculates the average Jaccard Similarity Index."""

        def calculate_jaccard_for_row(row):
            model_set = set(row[self.config.model_label_cols].dropna())
            clerical_set = set(row[self.config.clerical_label_cols].dropna())
            if not model_set and not clerical_set:
                return 1.0
            intersection = len(model_set.intersection(clerical_set))
            union = len(model_set.union(clerical_set))
            return intersection / union if union > 0 else 0.0

        return self.df.apply(calculate_jaccard_for_row, axis=1).mean()

    def get_accuracy(
        self, threshold: float = 0.0, match_type: str = "full", extended: bool = False
    ) -> Union[float, dict[str, float]]:
        """Calculate accuracy for predictions above a confidence threshold.

        Args:
            threshold (float): Minimum confidence score threshold.
            match_type (str): The type of accuracy to calculate.
                            Options: 'full' (default) or '2-digit'.
            extended (bool): If True, returns a dictionary with detailed accuracy metrics.
                            If False, returns only the accuracy percentage.

        Returns:
            Union[float, dict[str, float]]:
                - If extended is False: Accuracy as a percentage (float).
                - If extended is True: A dictionary containing:
                    - 'accuracy_percent' (float): Accuracy as a percentage.
                    - 'matches' (int): Number of matching predictions.
                    - 'non_matches' (int): Number of non-matching predictions.
                    - 'total_considered' (int): Total number of predictions considered.
        """
        # set a default return value:
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

        # 1. Filter the DataFrame based on the confidence threshold
        filtered_df = self.df[self.df["max_score"] >= threshold]
        total_in_subset = len(filtered_df)

        # Handle the edge case where no data meets the threshold
        if total_in_subset == 0:
            if extended:
                return {
                    "accuracy_percent": 0.0,
                    "matches": 0,
                    "non_matches": 0,
                    "total_considered": 0,
                }
            return 0.0

        # 2. Calculate the raw counts
        match_count = filtered_df[correct_col].sum()
        non_match_count = total_in_subset - match_count

        # 3. Calculate the percentage
        accuracy_percent = 100 * match_count / total_in_subset

        # 4. Return all values in a structured dictionary
        if extended:
            return_value = {
                "accuracy_percent": round(accuracy_percent, 1),
                "matches": int(match_count),
                "non_matches": int(non_match_count),
                "total_considered": total_in_subset,
            }
        else:
            return_value = accuracy_percent

        return return_value

    def get_candidate_contribution(self, candidate_col: str) -> dict[str, Any]:
        """Assesses the value add of a single candidate column using vectorised operations."""
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

    # ... other metric methods like get_coverage etc. would go here ...


# REFACTOR: This new class handles all plotting.
# It takes a DataFrame that has already been processed by the MetricCalculator.
class Visualizer:
    """Handles all visualization tasks."""

    def __init__(self, df: pd.DataFrame, calculator: MetricCalculator):
        """Initialises the Visualizer.

        Args:
            df (pd.DataFrame): The processed DataFrame with derived columns.
            calculator (MetricCalculator): An instance of the calculator to generate stats.
        """
        self.df = df
        self.calculator = calculator

    def plot_confusion_heatmap(
        self,
        matrix_config: ConfusionMatrixConfig,
        plot_config: Optional[PlotConfig] = None,
    ):
        """Generates a confusion matrix heatmap.

        Args:
            matrix_config (ConfusionMatrixConfig): Config specifying columns and top_n.
            plot_config (PlotConfig, optional): Config for saving and styling.
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
            return

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


class LabelAccuracy:
    """Orchestrates the data cleaning, metric calculation, and visualization
    for coder alignment analysis.
    """

    def __init__(self, df: pd.DataFrame, column_config: ColumnConfig):
        """Initialises the full analysis pipeline.

        Args:
            df (pd.DataFrame): The raw input DataFrame.
            column_config (ColumnConfig): The configuration for the analysis.
        """
        # Step 1: Clean and prepare the data
        cleaner = DataCleaner(df, column_config)
        clean_df = cleaner.process()

        # Step 2: Initialise the calculator with the clean data
        self.calculator = MetricCalculator(clean_df, column_config)

        # The final, processed DataFrame is stored here for inspection
        self.df = self.calculator.df

        # Step 3: Initialise the visualiser
        self.visualizer = Visualizer(self.df, self.calculator)

    # REFACTOR: Public methods now delegate their calls to the appropriate helper class.
    # This makes the LabelAccuracy class very simple and easy to read.
    def get_accuracy(self, **kwargs):
        """Calculate accuracy for predictions above a confidence threshold.

        Args:
            **kwargs: Arguments to pass to the calculator's get_accuracy method.

        Returns:
            The accuracy metric.
        """
        return self.calculator.get_accuracy(**kwargs)

    def get_jaccard_similarity(self, **kwargs):
        """Calculates the average Jaccard Similarity Index.

        Args:
            **kwargs: Arguments to pass to the calculator's get_jaccard_similarity.

        Returns:
            The Jaccard similarity score.
        """
        return self.calculator.get_jaccard_similarity(**kwargs)

    def plot_confusion_heatmap(self, **kwargs):
        """Generates and displays a confusion matrix heatmap.

        Args:
            **kwargs: Arguments to pass to the visualizer's plot_confusion_heatmap.
        """
        return self.visualizer.plot_confusion_heatmap(**kwargs)

    def get_candidate_contribution(self, **kwargs):
        """

        Args:
            **kwargs: Arguments to pass to the calculator's get_candidate_contribution.
        """
        return self.calculator.get_candidate_contribution(**kwargs)
                

    # ... you would add similar passthrough methods for all other public functions ...
    # e.g., get_coverage, get_summary_stats, plot_threshold_curves etc.

    @staticmethod
    def save_output(metadata: dict, eval_result: dict, save_path: str = "data/") -> str:
        """Save evaluation results to files.

        Args:
            metadata (dict): Dictionary of metadata parameters.
            eval_result (dict): Dictionary containing evaluation metrics.
            save_path (str): The folder where results should be saved.

        Returns:
            str: The folder path where results were stored.
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
