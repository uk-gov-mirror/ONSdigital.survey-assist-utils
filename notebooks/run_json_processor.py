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
"""Runs to test JsonProcessor."""
from typing import TypedDict

import pandas as pd
import toml
from google.cloud import storage
from IPython.display import Markdown, display

from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    ConfusionMatrixConfig,
    LabelAccuracy,
    PlotConfig,
)
from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor

# %% [markdown]
#
# This script includes all the metrics required for the PPT presentation as follows:
#
# 1) Observations in labelled set.
# 2) Variability across SIC sections.
# 3) Match of top CC vs top SA on unambiguously Codable - 2 digis, 5 digit.
# 4) Match of top CC vs any SA on unambiguously codable - 2 digis, 5 digit.
# 5) Match rate top CC vs any SA on all data - 2 digis, 5 digit.
# 6) Match rate any CC vs any SA on all data - 2 digis, 5 digit.
# 7) Jaccard's Score all CC vs all SA, all data - 2 digis, 5 digit.

# %%


# Define the metrics to run:
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


test_cases: list[TestCase] = [
    {
        "Test": "Match of top CC vs top SA on unambiguously Codable",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": True,
    },  # 3) Match of top CC vs top SA on unambiguously Codable - 2 digis, 5 digit.
    {
        "Test": "Match of top CC vs any SA on unambiguously codable",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": True,
    },  # 4) Match of top CC vs any SA on unambiguously codable - 2 digis, 5 digit.
    {
        "Test": "Match rate top CC vs any SA on all data",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": False,
    },  # 5) Match rate top CC vs any SA on all data - 2 digis, 5 digit.
    {
        "Test": "Match rate any CC vs any SA on all data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },  # 6) Match rate any CC vs any SA on all data - 2 digis, 5 digit.
    {
        "Test": "Jaccard's Score all CC vs all SA, all data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },  # 7) Jaccard's Score all CC vs all SA, all data - 2 digis, 5 digit.
]


# %%


def get_most_recent_gcs_file(bucket_name: str, prefix: str = "") -> str | None:
    """Finds the most recently modified file in a GCS bucket within a specific
    directory, excluding sub-directories.

    Args:
        bucket_name (str): The name of the GCS bucket (e.g., "my-data-bucket").
        prefix (str): The folder path to search within (e.g., "outputs/json_runs/").
                      If omitted, it searches the root of the bucket.
                      Ensure it ends with a '/' to specify a directory.

    Returns:
        str | None: The full gs:// path of the most recent file, or None if no
                    files are found in the specified location.
    """
    # Ensure the prefix ends with a slash to correctly identify it as a directory
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    storage_client = storage.Client()

    # Use delimiter='/' to prevent recursion into sub-directories.
    # This treats the bucket like a file system directory.
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix, delimiter="/")

    most_recent_blob = None
    latest_time = None

    # Iterate through all files at the specified prefix level to find the one
    # with the latest update time.
    for blob in blobs:
        # A blob's name can be the same as the prefix if the "folder" itself
        # is listed. We want to skip these and only process actual files.
        if blob.name == prefix:
            continue

        if latest_time is None or blob.updated > latest_time:
            latest_time = blob.updated
            most_recent_blob = blob

    if most_recent_blob:
        return f"gs://{bucket_name}/{most_recent_blob.name}"

    return None


# %%
with open("prepare_config.toml", encoding="utf-8") as file:
    config = toml.load(file)

# %%
MY_BUCKET_NAME = config["paths"]["gcs_bucket_name"]
after_docker_run = config["paths"]["gcs_json_dir"]

json_runs = get_most_recent_gcs_file(MY_BUCKET_NAME, prefix=after_docker_run)
print("JSON outputs in this file:", json_runs)


# %%
# This script relies on
# %run ../scripts/prepare_evaluation_data_for_analysis.py

# %%
# Write to the config the path of the processed dataframe
config["paths"]["processed_csv_output"] = config["paths"]["analysis_csv"]
# This file will be merged in to the json dataframe later
preprocessor = JsonPreprocessor(config)

# %%
record_count = preprocessor.count_all_records()
print("record_count", record_count)  # we expect 2079

llm_processed_df = preprocessor.process_files()
print("llm_processed_df shape", llm_processed_df.shape)

# %%
# Now merge the two dataframes:
full_output_df = preprocessor.merge_eval_data(llm_processed_df)
print("full_output_df shape", full_output_df.shape)

# %%
merged_file = config["paths"]["merged_file"]
print("merged_file ", merged_file)
full_output_df.to_csv(merged_file)


# %%

model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]


if full_output_df is not None:
    for case in test_cases:
        CC_COUNT = case["CCs"][0]
        LLM_COUNT = case["LLMs"][0]
        config_main = ColumnConfig(
            model_label_cols=model_label_cols[:LLM_COUNT],
            model_score_cols=model_score_cols[:LLM_COUNT],
            clerical_label_cols=clerical_label_cols[:CC_COUNT],
            id_col="unique_id",
            filter_unambiguous=case["Unambiguous"],
        )
        print(case["Test"])

        # Initialize the analyzer with the subset and config
        analyzer_main = LabelAccuracy(df=full_output_df, column_config=config_main)

        full_acc_stats = analyzer_main.get_accuracy(match_type="full", extended=True)
        if not isinstance(full_acc_stats, dict):
            raise TypeError(
                "Expected a dictionary from get_accuracy when extended=True"
            )
        print("full_acc_stats", full_acc_stats)

        digit_acc_stats = analyzer_main.get_accuracy(
            match_type="2-digit", extended=True
        )
        if not isinstance(digit_acc_stats, dict):
            raise TypeError(
                "Expected a dictionary from get_accuracy when extended=True"
            )
        print("digit_acc_stats", digit_acc_stats)


jaccard_results = analyzer_main.get_jaccard_similarity()
print("jaccard_results", jaccard_results)

jaccard_2_digit_score = analyzer_main.get_jaccard_similarity(match_type="2-digit")
print("jaccard_2_digit_score", jaccard_2_digit_score)

analyzer_main.plot_threshold_curves()

matrix_conf = ConfusionMatrixConfig(
    human_code_col=clerical_label_cols[0],
    llm_code_col=model_label_cols[0],
    exclude_patterns=["x", "-9", "4+"],
)

plot_conf = PlotConfig(save=False)
analyzer_main.plot_confusion_heatmap(matrix_config=matrix_conf, plot_config=plot_conf)

display(Markdown("### Summary Statistics Dictionary"))
summary_stats = analyzer_main.get_summary_stats()
display(pd.Series(summary_stats, name="Value").to_frame())
