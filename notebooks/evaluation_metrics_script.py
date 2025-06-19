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

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy

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
    test_directory = Path("/home/user/survey-assist-utils/data/artificial_data")
    file_test_list = [
        "unit_test_confidence.csv",
        "unit_test_coverage.csv",
        "unit_test_digits_accuracy.csv",
        "unit_test_label_accuracy.csv",
        "unit_test_heat_map.csv",
    ]

    # Select the file under test

    # Construct the full path to the file
    file_path = test_directory / file_test_list[4]

    # Read the CSV file with all data as strings
    process_data = pd.read_csv(file_path, dtype=str)
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
    process_data = pd.read_csv(
        "/home/user/survey-assist-utils/data/evaluation_data/combined_outputs.csv",
        dtype=str,
    )

print("process_data", process_data.shape)

analyzer = LabelAccuracy(df=process_data, column_config=col_config)

# %%
# Basic accuracy
digit_list = ["2-digit", "full"]
TYPE_DIGITS = digit_list[0]
print(
    f"Overall accuracy {TYPE_DIGITS},  {analyzer.get_accuracy(match_type = TYPE_DIGITS):.1f}%"
)
TYPE_DIGITS = digit_list[1]
print(
    f"Overall accuracy {TYPE_DIGITS},  {analyzer.get_accuracy(match_type = TYPE_DIGITS):.1f}%"
)

# %%
analyzer.get_coverage()

# %%
analyzer.get_threshold_stats()

# %%
analyzer.plot_threshold_curves()

# %%
analyzer.get_summary_stats()

# %%
human_code_col = col_config.clerical_label_cols[0]
llm_code_col = col_config.model_label_cols[0]
exclude_patterns = ["x", "-9"]
analyzer.plot_confusion_heatmap(
    human_code_col=human_code_col,
    llm_code_col=llm_code_col,
    top_n=10,
    exclude_patterns=exclude_patterns,
)
