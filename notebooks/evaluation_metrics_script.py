# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: survey-assist-utils-PWI-TvqZ-py3.12
#     language: python
#     name: python3
# ---

# %%
"""Runner script to process data through the coder_alignment."""


# %% [markdown]
# # Evaluation Matrix Runner
#
# This notebook runs a matrix of test scenarios, collates the numerical results,
# and saves them to a summary CSV file.

# %%
# --- Imports and Setup ---
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, Union

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy
from survey_assist_utils.logging import get_logger

# Create a logger instance
logger = get_logger(__name__)


# %%
def display_analysis_results(
    file_path: Path, test_description: str, analyzer: LabelAccuracy
):
    """Displays the results of an analysis in a structured Jupyter format.
    This version correctly handles the dictionary returned by get_accuracy.

    Args:
        file_path (Path): The full path to the test CSV file.
        test_description (str): The description of the evaluation being run.
        analyzer: An initialized LabelAccuracy instance.
    """
    # --- 1. Header and Data Loading Info ---
    display(Markdown(f"--- \n## Analysis for: `{file_path.name}`"))
    display(Markdown(f"**Test Scenario:** {test_description}"))
    display(Markdown(f"**Shape:** {analyzer.df.shape}"))

    # --- 2. Key Metrics (The Updated Section) ---
    display(Markdown("### Key Accuracy Metrics"))

    # Call the methods to get the full statistics dictionaries
    full_acc_stats = analyzer.get_accuracy(match_type="full", extended=True)
    if not isinstance(full_acc_stats, dict):
        raise TypeError("Expected a dictionary from get_accuracy when extended=True")

    digit_acc_stats = analyzer.get_accuracy(match_type="2-digit", extended=True)
    if not isinstance(digit_acc_stats, dict):
        raise TypeError("Expected a dictionary from get_accuracy when extended=True")

    # Display key metrics, now including the raw counts for verification
    key_metrics_df = pd.Series(
        {
            "Overall Accuracy (Full Match)": f"{full_acc_stats['accuracy_percent']:.1f}%",
            "Full Match Count": f"""{full_acc_stats['matches']} /
            {full_acc_stats['total_considered']}""",
            "Overall Accuracy (2-Digit Match)": f"{digit_acc_stats['accuracy_percent']:.1f}%",
            "2-Digit Match Count": f"""{digit_acc_stats['matches']} /
            {digit_acc_stats['total_considered']}""",
            "Overall Coverage": f"{analyzer.get_coverage():.1f}%",
        }
    ).to_frame("Value")
    display(key_metrics_df)

    # --- 3. Threshold Statistics (Unchanged) ---
    display(Markdown("### Accuracy/Coverage vs. Threshold"))
    threshold_stats = analyzer.get_threshold_stats()
    display(threshold_stats.head())

    # --- 4. Plots (Unchanged) ---
    display(Markdown("### Visualizations"))
    analyzer.plot_threshold_curves()
    analyzer.plot_confusion_heatmap(
        human_code_col=analyzer.config.clerical_label_cols[0],
        llm_code_col=analyzer.config.model_label_cols[0],
        top_n=10,
        exclude_patterns=["x", "-9"],
    )

    # --- 5. Summary Statistics (Unchanged) ---
    display(Markdown("### Summary Statistics Dictionary"))
    summary_stats = analyzer.get_summary_stats()
    display(pd.Series(summary_stats, name="Value").to_frame())


# %%
def calculate_analysis_metrics(analyzer: LabelAccuracy) -> dict[str, Any]:
    """Runs a suite of analyses and returns the results in a dictionary,
    including detailed match/non-match counts.

    Args:
        analyzer: An initialized LabelAccuracy instance.

    Returns:
        A dictionary containing all calculated metrics for one row of the results matrix.
    """
    # Call the methods to get the full statistics dictionaries
    full_acc_stats = analyzer.get_accuracy(match_type="full", extended=True)
    if not isinstance(full_acc_stats, dict):
        raise TypeError("Expected a dictionary from get_accuracy when extended=True")

    digit_acc_stats = analyzer.get_accuracy(match_type="2-digit", extended=True)
    if not isinstance(digit_acc_stats, dict):
        raise TypeError("Expected a dictionary from get_accuracy when extended=True")

    # Get the overall coverage (this can remain as is)
    coverage = analyzer.get_coverage()

    # Get the detailed summary stats dictionary
    summary_stats = analyzer.get_summary_stats()

    # 2. Combine all results into a single, detailed dictionary
    # Unpack the dictionaries from get_accuracy with descriptive prefixes
    results = {
        # --- Full Match Stats ---
        "accuracy_full_match_percent": full_acc_stats["accuracy_percent"],
        "full_match_count": full_acc_stats["matches"],
        "full_match_unmatched": full_acc_stats["non_matches"],
        "full_match_total_considered": full_acc_stats["total_considered"],
        # --- 2-Digit Match Stats ---
        "accuracy_2_digit_match_percent": digit_acc_stats["accuracy_percent"],
        "2_digit_match_count": digit_acc_stats["matches"],
        "2_digit_match_unmatched": digit_acc_stats["non_matches"],
        "2_digit_match_total_considered": digit_acc_stats["total_considered"],
        # --- Other Stats ---
        "overall_coverage_percent": f"{coverage:.1f}",
        **summary_stats,  # Unpack the rest of the summary stats
    }
    return results


# %%
def load_main_dataframe(
    path: Path, columns_to_clean: list
) -> Union[pd.DataFrame, None]:
    """Attempts to load a CSV file into a pandas DataFrame.

    Logs success or failure, and returns the DataFrame if successful,
    or None if the file is not found.

    Args:
        path (Path): The full path to the CSV file.
        columns_to_clean (list): columns to replace imposter nans

    Returns:
        Union[pd.DataFrame, None]: The loaded DataFrame, or None if the file is missing.
    """
    missing_value_formats = ["", " ", "nan", "None", "Null", "<NA>"]
    try:
        df = pd.read_csv(path, dtype=str)
        df[columns_to_clean] = df[columns_to_clean].replace(
            missing_value_formats, np.nan
        )

        logger.info(f"Successfully loaded main data file with shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Main data file not found: {path}")
        return None


# %%
# --- Step 3: Define the Test Matrix and Configuration ---


class TestCase(TypedDict):
    """Represents a single test case configuration for evaluating model behavior.

    Attributes:
        Test (str): A descriptive label or name for the test scenario.
        CCs (list[int]): A list of content classifier (CC) identifiers used in the test.
        LLMs (list[int]): A list of large language model (LLM) identifiers used in the test.
        Unambiguous (bool): Indicates whether the test scenario is unambiguous (True) or
        ambiguous (False).
    """

    Test: str
    CCs: list[int]
    LLMs: list[int]
    Unambiguous: bool


file_directory = Path("/home/user/survey-assist-utils/data/evaluation_data/")
FILE_TO_TEST = "DSC_Rep_Sample_test_end_to_end_20250620_142222_output.csv"
# FILE_TO_TEST = "merged_df_confirmed.csv"
full_file_path = file_directory / FILE_TO_TEST

model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

test_cases: list[TestCase] = [
    {
        "Test": "Top SA vs Top CC, All Data",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": False,
    },
    {
        "Test": "Top SA vs Top CC, Unambiguous",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": True,
    },
    {
        "Test": "Any of 5 SA vs Any of 3 CC, All Data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },
    {
        "Test": "Any of 5 SA vs Any of 3 CC, Unambiguous",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": True,
    },
    {
        "Test": "Any of 5 SA vs Top CC, All Data",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": False,
    },
    {
        "Test": "Any of 5 SA vs Top CC, Unambiguous",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": True,
    },
]

# --- Step 4: Run the Matrix, Collect Results, and Save to CSV ---
columns_to_clean_main = model_label_cols + model_score_cols + clerical_label_cols
main_dataframe = load_main_dataframe(full_file_path, columns_to_clean_main)

all_results = []

if main_dataframe is not None:
    for case in test_cases:
        for cc_count in case["CCs"]:
            for llm_count in case["LLMs"]:
                # Create the specific configuration for this test run
                config_main = ColumnConfig(
                    model_label_cols=model_label_cols[:llm_count],
                    model_score_cols=model_score_cols[:llm_count],
                    clerical_label_cols=clerical_label_cols[:cc_count],
                    id_col="unique_id",
                    filter_unambiguous=case["Unambiguous"],
                )

                # Initialize the analyzer with the data and config
                analyzer_main = LabelAccuracy(
                    df=main_dataframe.copy(), column_config=config_main
                )

                # ---- A. Calculate the metrics ----
                test_results = calculate_analysis_metrics(analyzer_main)

                # ---- B. Display the visuals (optional) ----
                display_analysis_results(
                    file_path=full_file_path,
                    test_description=case["Test"],
                    analyzer=analyzer_main,
                )

                # ---- C. Collate the results for the final CSV ----
                # Add the test case description to the results dictionary
                record = {"test_scenario": case["Test"], **test_results}
                all_results.append(record)

# --- Step 5: Save the Collated Results ---

if all_results:
    # Convert the list of dictionaries into a DataFrame
    results_df = pd.DataFrame(all_results)

    # Create a timestamped filename for the output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUE_FILENAME = f"evaluation_summary_{timestamp}.csv"
    output_path = file_directory / OUTPUE_FILENAME

    # Save to CSV
    results_df.to_csv(output_path, index=False)

    logger.info(f" Successfully saved evaluation summary to: {output_path}")
    display(Markdown("###  Final Collated Results"))
    display(results_df)

# %% [markdown]
# ###  Add in matrix for 2 digits
