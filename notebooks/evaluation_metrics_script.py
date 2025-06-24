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
"""Temporary runner script to test the coder_alignment."""


# %%
from pathlib import Path
from typing import TypedDict

import pandas as pd
from IPython.display import Markdown, display

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy
from survey_assist_utils.logging import get_logger

# Create a logger instance
logger = get_logger(__name__)


# %%
def run_and_display_analysis(
    file_path: Path, config: ColumnConfig, test_description: str
):
    """Loads a single test file, runs a full suite of analyses, and displays
    the results in a structured and readable format in a Jupyter notebook.

    Args:
        file_path (Path): The full path to the test CSV file.
        config (ColumnConfig): The configuration object defining column names.
        test_description (str): The description of the evaluatoin being run.
    """
    # --- 1. Header and Data Loading ---
    # Use Markdown for a header for each file's section
    display(Markdown(f"{test_description}--- \n## 📊 Analysis for: `{file_path.name}`"))

    try:
        process_data = pd.read_csv(file_path, dtype=str)
        logger.info(f"Successfully loaded data with shape: {process_data.shape}")
        display(Markdown(f"**Shape:** {process_data.shape}"))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return
    print(f"Successfully loaded data with shape: {process_data.shape}")
    analyzer = LabelAccuracy(df=process_data, column_config=config)

    # --- 2. Key Metrics ---
    display(Markdown("### Key Accuracy Metrics"))
    full_acc = analyzer.get_accuracy(match_type="full")
    digit_acc = analyzer.get_accuracy(match_type="2-digit")

    # Display key metrics in a small table for clarity
    key_metrics = pd.Series(
        {
            "Overall Accuracy (Full Match)": f"{full_acc:.1f}%",
            "Overall Accuracy (2-Digit Match)": f"{digit_acc:.1f}%",
            "Overall Coverage": f"{analyzer.get_coverage():.1f}%",
        }
    ).to_frame("Value")
    display(key_metrics)

    # --- 3. Threshold Statistics ---
    display(Markdown("### Accuracy/Coverage vs. Threshold"))
    threshold_stats = analyzer.get_threshold_stats()
    display(threshold_stats.head())  # Display the head of the stats table

    # --- 4. Plots ---
    display(Markdown("### Visualizations"))
    analyzer.plot_threshold_curves()

    # Set up columns for the confusion matrix from the config
    human_code_col = config.clerical_label_cols[0]
    llm_code_col = config.model_label_cols[0]

    analyzer.plot_confusion_heatmap(
        human_code_col=human_code_col,
        llm_code_col=llm_code_col,
        top_n=10,
        exclude_patterns=["x", "-9"],
    )

    # --- 5. Summary Statistics ---
    display(Markdown("### Summary Statistics Dictionary"))
    summary_stats = analyzer.get_summary_stats()
    # Convert the dictionary to a pandas Series for a nice table display
    display(pd.Series(summary_stats, name="Value").to_frame())


# %%
# Setup for a test case
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
file_test_list = ["DSC_Rep_Sample_test_end_to_end_20250620_142222_output.csv"]

model_label_cols = [
    "candidate_1_sic_code",
    "candidate_2_sic_code",
    "candidate_3_sic_code",
    "candidate_4_sic_code",
    "candidate_5_sic_code",
]

model_score_cols = [
    "candidate_1_likelihood",
    "candidate_2_likelihood",
    "candidate_3_likelihood",
    "candidate_4_likelihood",
    "candidate_5_likelihood",
]

clerical_label_cols = ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]

# Define the test matrix
test_cases: list[TestCase] = [
    {
        "Test": "Scenario 1: Top SA & Top CC, All data",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": False,
    },
    {
        "Test": "Scenario 1: Top SA & Top CC, All data",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": False,
    },
    {
        "Test": "Scenario 1: Top SA & Top CC, Unambiguous only",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": True,
    },
    {
        "Test": "Scenario 2: SA match any CC, all data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },
    {
        "Test": "Scenario 2: SA match any CC, Unambiguous only",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": True,
    },
    {
        "Test": "Scenario 3: Do any SA match top CC, all data",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": False,
    },
    {
        "Test": "Scenario 3: Do any SA match top CC, Unambiguous only",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": True,
    },
]

# Loop through each test case and call the function
for case in test_cases:
    for cc in case["CCs"]:
        for llm in case["LLMs"]:
            col_config = ColumnConfig(
                model_label_cols=model_label_cols[0:llm],
                model_score_cols=model_score_cols[0:llm],
                clerical_label_cols=clerical_label_cols[0:cc],
                id_col="unique_id",
                filter_unambiguous=case["Unambiguous"],
            )
            for filename in file_test_list:
                full_path = file_directory / filename
                run_and_display_analysis(
                    file_path=full_path,
                    config=col_config,
                    test_description=case["Test"],
                )
