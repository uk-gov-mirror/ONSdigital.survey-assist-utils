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

import pandas as pd
from IPython.display import Markdown, display

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy
from survey_assist_utils.logging import get_logger

# Create a logger instance
logger = get_logger(__name__)


# %%


# %%
def run_and_display_analysis(file_path: Path, config: ColumnConfig):
    """Loads a single test file, runs a full suite of analyses, and displays
    the results in a structured and readable format in a Jupyter notebook.

    Args:
        file_path (Path): The full path to the test CSV file.
        config (ColumnConfig): The configuration object defining column names.
    """
    # --- 1. Header and Data Loading ---
    # Use Markdown for a clean, bold header for each file's section
    display(Markdown(f"--- \n## 📊 Analysis for: `{file_path.name}`"))

    try:
        process_data = pd.read_csv(file_path, dtype=str)
        logger.info(f"Successfully loaded data with shape: {process_data.shape}")
        display(Markdown(f"**Shape:** {process_data.shape}"))
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return

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
# Set up some lists to hold the columns we are interested in:
TEST_ENVIRONMENT = True
if TEST_ENVIRONMENT:
    col_config = ColumnConfig(
        clerical_label_cols=["CC_1", "CC_2", "CC_3"],
        model_label_cols=["SA_1", "SA_2", "SA_3", "SA_4", "SA_5"],
        model_score_cols=[
            "SA_score_1",
            "SA_score_2",
            "SA_score_3",
            "SA_score_4",
            "SA_score_5",
        ],
        id_col="unique_id",
    )

    # Define the directory and file list
    file_directory = Path("/home/user/survey-assist-utils/data/artificial_data")
    file_test_list = [
        # "unit_test_confidence.csv",
        # "unit_test_coverage.csv",
        # "unit_test_digits_accuracy.csv",
        # "unit_test_label_accuracy.csv",
        # "unit_test_heat_map.csv",
        "unit_test_not_first_choice.csv"
    ]

else:
    col_config = ColumnConfig(
        model_label_cols=[
            "candidate_1_sic_code",
            "candidate_2_sic_code",
            "candidate_3_sic_code",
            "candidate_4_sic_code",
            "candidate_5_sic_code",
        ],
        model_score_cols=[
            "candidate_1_likelihood",
            "candidate_2_likelihood",
            "candidate_3_likelihood",
            "candidate_4_likelihood",
            "candidate_5_likelihood",
        ],
        clerical_label_cols=["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"],
        id_col="unique_id",
    )
    file_directory = Path("/home/user/survey-assist-utils/data/evaluation_data")
    file_test_list = ["combined_outputs.csv"]

# %%
# Loop through each file and run the complete analysis
for filename in file_test_list:
    full_path = file_directory / filename
    run_and_display_analysis(file_path=full_path, config=col_config)

# %%
# How well does our first choice match the CCs first choice?
col_config_top_choice_only = ColumnConfig(
    clerical_label_cols=["CC_1"],
    model_label_cols=["SA_1"],
    model_score_cols=["SA_score_1"],
    id_col="unique_id",
)

# Loop through each file and run the complete analysis
for filename in file_test_list:
    full_path = file_directory / filename
    run_and_display_analysis(file_path=full_path, config=col_config_top_choice_only)

# %%
# Is the CC's first choice found somewhere in our suggestions?
col_config_first_in_set = ColumnConfig(
    clerical_label_cols=["CC_1"],
    model_label_cols=["SA_1", "SA_2", "SA_3", "SA_4", "SA_5"],
    model_score_cols=[
        "SA_score_1",
        "SA_score_2",
        "SA_score_3",
        "SA_score_4",
        "SA_score_5",
    ],
    id_col="unique_id",
)

# Loop through each file and run the complete analysis
for filename in file_test_list:
    full_path = file_directory / filename
    run_and_display_analysis(file_path=full_path, config=col_config_first_in_set)
