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

import importlib

# %%
from pathlib import Path

import pandas as pd

from survey_assist_utils.evaluation import coder_alignment
from survey_assist_utils.evaluation.coder_alignment import LabelAccuracy

importlib.reload(coder_alignment)

# %%
# Set up some lists to hold the columns we are interested in:
test_environment = True

if test_environment:
    coders_list = ["CC_1", "CC_2", "CC_3"]
    candidate_list = ["SA_1", "SA_2", "SA_3", "SA_4", "SA_5"]
    candidate_likelihood = [
        "SA_score_1",
        "SA_score_2",
        "SA_score_3",
        "SA_score_4",
        "SA_score_5",
    ]

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
    file_under_test = file_test_list[4]

    # Construct the full path to the file
    file_path = test_directory / file_under_test

    # Read the CSV file with all data as strings
    process_data = pd.read_csv(file_path, dtype=str)
else:
    candidate_list = [
        "candidate_1_sic_code",
        "candidate_2_sic_code",
        "candidate_3_sic_code",
        "candidate_4_sic_code",
        "candidate_5_sic_code",
    ]
    candidate_likelihood = [
        "candidate_1_likelihood",
        "candidate_2_likelihood",
        "candidate_3_likelihood",
        "candidate_4_likelihood",
        "candidate_5_likelihood",
    ]
    coders_list = ["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"]
    process_data = pd.read_csv(
        "/home/user/survey-assist-utils/data/evaluation_data/combined_outputs.csv",
        dtype=str,
    )

print("process_data", process_data.shape)

# %%
# Create the object to work on with the analysis:
analyzer = LabelAccuracy(
    process_data,
    id_col="unique_id",
    model_label_cols=candidate_list,
    model_score_cols=candidate_likelihood,
    clerical_label_cols=coders_list,
)

# %%
# Basic accuracy
digit_list = ["2-digit", "full"]
type_digits = digit_list[0]
print(
    f"Overall accuracy {type_digits},  {analyzer.get_accuracy(threshold = 0.0, match_type = type_digits):.1f}%"
)
type_digits = digit_list[1]
print(
    f"Overall accuracy {type_digits},  {analyzer.get_accuracy(threshold = 0.0, match_type = type_digits):.1f}%"
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
human_code_col = coders_list[0]
llm_code_col = candidate_list[0]
exclude_patterns = ["x", "-9"]
analyzer.plot_confusion_heatmap(
    human_code_col=human_code_col,
    llm_code_col=llm_code_col,
    top_n=10,
    exclude_patterns=exclude_patterns,
)
