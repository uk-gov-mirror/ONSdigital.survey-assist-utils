"""Module contains functionality to evaluate alignment between Clerical Coders (CC)
and Survey Assist (SA) results.

The classes are:
ColumnConfig
    A data structure to hold the name configurations for the analysis.

LabelAccuracy
    Analyse classification accuracy for scenarios where model predictions can match any of
    multiple ground truth labels.

The methods are:
_melt_and_clean
    A helper function to reshape, clean, and prepare data for matching.
_add_derived_columns
    Adds computed columns for full and partial matches.
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
from typing import Optional, Union, Set, Any

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

    def _melt_and_clean(self, value_vars: list[str], value_name: str) -> pd.DataFrame:
        """A helper function to reshape, clean, and prepare data for matching.

        It takes a list of columns, melts them into a long format, replaces
        various missing value formats with a standard NaN, and drops them.

        Args:
            value_vars (list[str]): The columns to melt (e.g., model or clerical cols).
            value_name (str): The name to give the new value column (e.g., 'model_label').

        Returns:
            pd.DataFrame: A cleaned, long-format DataFrame with id and value columns.
        """
        missing_value_formats = ["", " ", "nan", "None", "Null", "<NA>"]

        # Melt the specified columns
        melted_df = self.df.melt(
            id_vars=[self.id_col],
            value_vars=value_vars,
            value_name=value_name,
        )

        # Replace all non-standard missing values and drop them in one chain
        melted_df[value_name] = melted_df[value_name].replace(
            missing_value_formats, np.nan
        )
        cleaned_df = melted_df.dropna(subset=[value_name])

        return cleaned_df

    def _add_derived_columns(self):
        """Adds computed columns for full and partial matches (vectorized)."""
        # --- Step 1: Reshape data using the cleaner function ---
        model_melted = self._melt_and_clean(
            value_vars=self.model_label_cols, value_name="model_label"
        )
        clerical_melted = self._melt_and_clean(
            value_vars=self.clerical_label_cols, value_name="clerical_label"
        )

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



    def _get_sets_from_row(self, row: pd.Series) -> tuple[Set[Any], Set[Any]]:
        """
        A private helper to extract unique, non-null sets of labels from a row.

        Args:
            row (pd.Series): A single row of the DataFrame.

        Returns:
            tuple[Set[Any], Set[Any]]: A tuple containing two sets:
                                       the model labels and the clerical labels.
        """
        # Get unique, non-null values from the model prediction columns
        model_labels = set(pd.Series(row[self.model_label_cols]).dropna().unique())
        
        # Get unique, non-null values from the ground truth columns
        clerical_labels = set(pd.Series(row[self.clerical_label_cols]).dropna().unique())
        
        return model_labels, clerical_labels

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
        # Set a fefault return value:

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


    def get_jaccard_similarity(self) -> float:
        """
        Calculates the Jaccard Similarity Index for each row and returns the average.

        For each row, it compares the set of model-predicted labels with the set
        of clerical (ground truth) labels. The Jaccard Index is the size of
        the intersection divided by the size of the union of these two sets.

        Returns:
            float: The average Jaccard Similarity Index across all rows in the
                   DataFrame, returned as a percentage.
        """
        
        def calculate_jaccard_for_row(row):
            model_set, clerical_set = self._get_sets_from_row(row)

            # If either set is empty, the union will not be empty if the other
            # set has items. An empty intersection results in a score of 0.
            if not model_set and not clerical_set:
                return 1.0  # Both sets are empty, perfect agreement.
            
            intersection_size = len(model_set.intersection(clerical_set))
            union_size = len(model_set.union(clerical_set))
            
            if union_size == 0:
                return 1.0 # Technically covered by the check above, but safe.
                
            return intersection_size / union_size

        # Apply the calculation to every row of the DataFrame
        jaccard_scores = self.df.apply(calculate_jaccard_for_row, axis=1)
        
        # Return the average score across all rows, as a percentage
        average_jaccard_percentage = jaccard_scores.mean() * 100
        
        return round(average_jaccard_percentage, 2)



    def get_candidate_contribution(self, candidate_col: str) -> dict[str, Any]:
        """
        Assesses the value add of a single candidate column.

        This method calculates how often a specific candidate's prediction matches
        any of the ground truth labels, and also how often it specifically
        matches the primary ground truth label.

        Args:
            candidate_col (str): The name of the candidate column to evaluate
                                 (e.g., 'candidate_5_sic_code').

        Returns:
            Dict[str, Any]: A dictionary containing the match rates and counts.
        """
        # --- 1. Validate that all necessary columns exist ---
        primary_clerical_col = self.clerical_label_cols[0]
        required_cols = [candidate_col, primary_clerical_col] + self.clerical_label_cols
        
        if any(col not in self.df.columns for col in required_cols):
            raise ValueError(f"One or more required columns not found for this analysis.")

        # Create a working copy, dropping rows where the candidate has no code
        # This ensures we only evaluate the candidate where it made a prediction.
        working_df = self.df[[candidate_col, *self.clerical_label_cols]].dropna(
            subset=[candidate_col]
        )
        total_considered = len(working_df)

        if total_considered == 0:
            return {
                "candidate_column": candidate_col,
                "total_predictions_made": 0,
                "primary_match_percent": 0.0,
                "primary_match_count": 0,
                "any_clerical_match_percent": 0.0,
                "any_clerical_match_count": 0,
            }

        # --- 2. Calculate match against the PRIMARY clerical code ---
        primary_matches_mask = (
            working_df[candidate_col] == working_df[primary_clerical_col]
        )
        primary_match_count = primary_matches_mask.sum()
        primary_match_percent = 100 * primary_match_count / total_considered

        # --- 3. Calculate match against ANY of the clerical codes ---
        def check_row_for_any_match(row):
            # Get the single value from the candidate column for this row
            candidate_value = row[candidate_col]
            # Get the list of all clerical codes for this row
            clerical_values = list(row[self.clerical_label_cols].dropna())
            return candidate_value in clerical_values

        any_match_mask = working_df.apply(check_row_for_any_match, axis=1)
        any_match_count = any_match_mask.sum()
        any_match_percent = 100 * any_match_count / total_considered

        # --- 4. Return results in a structured dictionary ---
        return {
            "candidate_column": candidate_col,
            "total_predictions_made": total_considered,
            "primary_match_percent": round(primary_match_percent, 2),
            "primary_match_count": int(primary_match_count),
            "any_clerical_match_percent": round(any_match_percent, 2),
            "any_clerical_match_count": int(any_match_count),
        }


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
