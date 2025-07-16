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

import pandas as pd
import toml
from google.cloud import storage
from IPython.display import Markdown, display

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy
from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor

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
print("record_count", record_count)  # we expect 10

llm_processed_df = preprocessor.process_files()
print("llm_processed_df shape", llm_processed_df.shape)

# %%
# Now merge the two dataframes:
full_output_df = preprocessor.merge_eval_data(llm_processed_df)
print("full_output_df shape", full_output_df.shape)

# %%
full_output_df.to_csv("full_output_df.csv")

# %%


model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

config_main = ColumnConfig(
    model_label_cols=model_label_cols,
    model_score_cols=model_score_cols,
    clerical_label_cols=clerical_label_cols,
    id_col="unique_id",
    filter_unambiguous=False,
)

analyzer = LabelAccuracy(df=full_output_df, column_config=config_main)

# %%

full_acc_stats = analyzer.get_accuracy(match_type="full", extended=True)
if not isinstance(full_acc_stats, dict):
    raise TypeError("Expected a dictionary from get_accuracy when extended=True")

digit_acc_stats = analyzer.get_accuracy(match_type="2-digit", extended=True)
if not isinstance(digit_acc_stats, dict):
    raise TypeError("Expected a dictionary from get_accuracy when extended=True")

# Get the overall coverage
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
print(results)

# %%
# Implement testing
analyzer.plot_threshold_curves()

# %%
analyzer.plot_confusion_heatmap(
    human_code_col=analyzer.config.clerical_label_cols[0],
    llm_code_col=analyzer.config.model_label_cols[0],
    top_n=10,
    exclude_patterns=["x", "-9"],
)

# %%

display(Markdown("### Summary Statistics Dictionary"))
summary_stats = analyzer.get_summary_stats()
display(pd.Series(summary_stats, name="Value").to_frame())
