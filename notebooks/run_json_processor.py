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
"""Runs to Call JsonProcessor and assess the recent LLM metrics."""
from typing import TypedDict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import toml
from IPython.display import Markdown, display

# gcloud auth application-default login
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
# For security, the actual bucket name has been changed to "\<my-butket-name\>"
#
# 1) Observations in labelled set.
# 2) Variability across SIC sections.
# 3) Match of top CC vs top SA on unambiguously Codable - 2 digis, 5 digit.
# 4) Match of top CC vs any SA on unambiguously codable - 2 digis, 5 digit.
# 5) Match rate top CC vs any SA on all data - 2 digis, 5 digit.
# 6) Match rate any CC vs any SA on all data - 2 digis, 5 digit.
# 7) Jaccard's Score all CC vs all SA, all data - 2 digis, 5 digit.
#
# We will first process the JSON output from the original 2000 data batch run.
#
# This requires a setup toml file, in 'prepare_config.toml' in this directory.
#
# It contains the following:
# [paths]
# * the original Title data is here:
# batch_filepath = "gs://<my-butket-name>/evaluation_data/DSC_Rep_Sample.csv"
# gcs_bucket_name = "<my-butket-name>"
# evaluation_data = "evaluation_data/"
#
# * The Docker run puts the data here
# gcs_json_dir = "analysis_outputs/"
#
# * To process a single  JSON file: set single_file = "True"
# named_file = "gs://<my-butket-name>/analysis_outputs/20250620_153641_output.json"
#
# * Otherwise the processor will assume all the JSON files after a
#  specified date are relevant to the evaluation
#
# * After running 'add_data_quality_flags', the helper columns will be
#  written to here:
# analysis_csv = "gs://<my-butket-name>/analysis_outputs/added_columns/flags_20250620_153641.csv"
#
#
# * To process a list of files that have already been merged with the input
# annotated CC data:
#
# merged_file_list = [
# "gs://<my-butket-name>/analysis_outputs/
# merged_files/merged_flags_20250620_153641.csv"
# "gs://<my-butket-name>/analysis_outputs/
# merged_files/Jyl_2025-06-27_codability_gemini_1.5-flash.csv"
# "gs://<my-butket-name>/analysis_outputs/
# merged_files/Jyl_2025-06-27_codability_gemini_2-flash.csv"
# ]
#
#
# [parameters]
# ### Process only files created on or after this date (YYYYMMDD), or only a specified file.
# single_file = "True"
# date_since = "20250710"

# %% [markdown]
# ### Next, we set up the metrics that we defined in the Power Point presentation,
#  taken from the evaluation plan:
#
#

# %%


# Define the metrics to run:
class EvaluationCase(TypedDict):
    """Represents a single Title case configuration for evaluating model behavior.

    Attributes:
        Title (str): A descriptive label or name for the Title scenario.
        CCs (list[int]): A list of content classifier (CC) identifiers used in the Title.
        LLMs (list[int]): A list of large language model (LLM) identifiers used in the Title.
        Unambiguous (bool): Indicates whether the Title scenario is unambiguous (True) or
        ambiguous (False).
    """

    Title: str
    CCs: list[int]
    LLMs: list[int]
    Unambiguous: bool


evaluation_cases_main: list[EvaluationCase] = [
    {
        "Title": "Match of top CC vs top SA on unambiguously Codable",
        "CCs": [1],
        "LLMs": [1],
        "Unambiguous": True,
    },
    # 3) Match of top CC vs top SA on unambiguously Codable - 2 digis, 5 digit.
    {
        "Title": "Match of top CC vs any SA on unambiguously codable",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": True,
    },
    # 4) Match of top CC vs any SA on unambiguously codable - 2 digis, 5 digit.
    {
        "Title": "Match rate top CC vs any SA on all data",
        "CCs": [1],
        "LLMs": [5],
        "Unambiguous": False,
    },
    # 5) Match rate top CC vs any SA on all data - 2 digis, 5 digit.
    {
        "Title": "Match rate any CC vs any SA on all data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },
    # 6) Match rate any CC vs any SA on all data - 2 digis, 5 digit.
    {
        "Title": "Jaccard's Score all CC vs all SA, all data",
        "CCs": [3],
        "LLMs": [5],
        "Unambiguous": False,
    },
    # 7) Jaccard's Score all CC vs all SA, all data - 2 digis, 5 digit.
]


# %%
with open("prepare_config.toml", encoding="utf-8") as file:
    config = toml.load(file)

# %%
MY_BUCKET_NAME = config["paths"]["gcs_bucket_name"]
after_docker_run = config["paths"]["gcs_json_dir"]


# %% [markdown]
# ### This preparation script only needs running once:
#
#

# %%
# This script relies on
# #%run ../scripts/prepare_evaluation_data_for_analysis.py

# %% [markdown]
# ### The config is set up to process the original JSON
# from the 2000 run and original prompt and model:
#
#

# %%
# Get a list of files to check:
preprocessor = JsonPreprocessor(config)
record_count = preprocessor.count_all_records()
print("record_count", record_count)  # we expect 2079

llm_processed_df = preprocessor.process_files()
print("llm_processed_df shape", llm_processed_df.shape)

# %% [markdown]
# ### We now have a flattened JSON file containing the model's responses to
# each situation from the data.
#
# Next we will run the evaluation from the first run:

# %% [markdown]
# ### We merge the flattened JSON with the annotated dataset
#
#
#

# %%
# Take the output from the preparation script and make it the input to the merging:
config["paths"]["processed_csv_output"] = config["paths"]["analysis_csv"]
full_output_df = preprocessor.merge_eval_data(llm_processed_df)


# %% [markdown]
# ### We check the input data distributions
#
#

# %%


def display_group_membership_analysis(df: pd.DataFrame):
    """Calculates and displays a pie chart and summary of group memberships."""
    total_count = len(df)
    having_four_responses = 4

    # Calculate counts for each category
    count_uncodeable = (df["num_answers"] == 0).sum()
    count_2_digits_only = df["Match_2_digits"].sum()
    count_5_digits_unambiguous = df["Unambiguous"].sum()
    count_5_digits_ambiguous = (df["Match_5_digits"] & ~df["Unambiguous"]).sum()
    count_3_digits_only = df["Match_3_digits"].sum()
    count_four_plus = (df["num_answers"] == having_four_responses).sum()

    group_counts = [
        count_uncodeable,
        count_four_plus,
        count_2_digits_only,
        count_3_digits_only,
        count_5_digits_unambiguous,
        count_5_digits_ambiguous,
    ]

    group_labels = [
        "Uncodeable",
        "4+ Codes",
        "Codeable to 2 digits only",
        "Codeable to 3 digits only",
        "Codeable unambiguously to 5 digits",
        "Codeable ambiguously to 5 digits",
    ]

    total_categorised = sum(group_counts)
    count_other = total_count - total_categorised

    if count_other > 0:
        group_counts.append(count_other)
        group_labels.append("Other")

    # Create and display the pie chart
    plt.figure(figsize=(10, 10))
    plt.pie(
        group_counts,
        autopct=lambda p: f"{p:.1f}%" if p > 1 else "",
        startangle=90,
        colors=sns.color_palette("pastel", len(group_counts)),
    )
    plt.title("Distribution of Group Membership in Labelled Data", fontsize=16)
    plt.legend(
        labels=[
            f"{label} ({count/total_count*100:.1f}%)"
            for label, count in zip(group_labels, group_counts)
        ],
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
    )
    plt.axis("equal")
    plt.show()
    plt.close()

    # Create and display the summary table
    percentages = [(count / total_count) * 100 for count in group_counts]
    summary_df = pd.DataFrame(
        {"Category": group_labels, "Count": group_counts, "Percentage": percentages}
    )
    display(summary_df)


display_group_membership_analysis(full_output_df)


# %% [markdown]
# ### And check the distribution of SIC Section membership:
#
#


# %%
def plot_sic_division_histogram(df: pd.DataFrame):
    """Generates and displays a histogram for the SIC_Division column."""
    plt.figure(figsize=(12, 8))

    counts = df["SIC_Division"].value_counts(normalize=True).nlargest(10) * 100

    ax = sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    ax.set_title(f"Top 10 Most Frequent SIC Divisions (Total Records: {len(df)})")
    plt.ylabel("Percentage")
    plt.xlabel("SIC Division (2-Digit Code)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    plt.close()


plot_sic_division_histogram(full_output_df)

# %% [markdown]
# ### results from the powerpoint slide 16:
#
# 2-digit SIC 	71%
# (division)
# 5-digit SIC 	55%
# (sub-class)
#


# %%
def all_results(df, evaluation_case):
    """Execute a full evaluation pipeline on a DataFrame using a series of
    predefined evaluation cases.

    For each evaluation case, this function:
    - Configures the label columns based on the number of LLM and clerical codes specified.
    - Initializes a `LabelAccuracy` analyzer with the configuration.
    - Computes and prints full and 2-digit match accuracy statistics.
    - Computes and prints Jaccard similarity scores (full and 2-digit).
    - Plots threshold performance curves.
    - Generates and displays a confusion matrix heatmap.
    - Displays summary statistics in a formatted table.

    Args:
        df (pd.DataFrame): The input DataFrame containing model-generated and
        clerical label columns, as well as a unique identifier column (`unique_id`).
        evaluation_case (list[dict]): A list of dictionaries, each defining an evaluation scenario.
            Each dictionary should include:
            - "LLMs": list[int] — number of model label columns to use.
            - "CCs": list[int] — number of clerical label columns to use.
            - "Unambiguous": bool — whether to filter ambiguous rows.
            - "Title": str — a label for the test case.

    Side Effects:
        - Prints evaluation metrics to the console.
        - Displays plots and summary statistics using IPython display tools.
        - Raises TypeError if `get_accuracy` does not return a
        dictionary when `extended=True`.

    Notes:
        - Assumes standard naming conventions for model and clerical label columns.
        - Uses the first model and clerical label columns for confusion matrix plotting.
    """
    # Set up standard column names
    model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
    model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
    clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

    if df is not None:
        for case in evaluation_case:
            config_main = ColumnConfig(
                model_label_cols=model_label_cols[: case["LLMs"][0]],
                model_score_cols=model_score_cols[: case["LLMs"][0]],
                clerical_label_cols=clerical_label_cols[: case["CCs"][0]],
                id_col="unique_id",
                filter_unambiguous=case["Unambiguous"],
            )
            print(case["Title"])

            # Initialize the analyzer with the subset and config
            analyzer_main = LabelAccuracy(df=df, column_config=config_main)

            full_acc_stats = analyzer_main.get_accuracy(
                match_type="full", extended=True
            )
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
            print("Two digit_acc_stats", digit_acc_stats)

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
    analyzer_main.plot_confusion_heatmap(
        matrix_config=matrix_conf, plot_config=plot_conf
    )

    display(Markdown("### Summary Statistics Dictionary"))
    summary_stats = analyzer_main.get_summary_stats()
    display(pd.Series(summary_stats, name="Value").to_frame())


# %%
# In the config file we have a list of merged and flattened files that we can process:
print(config["paths"]["merged_file_list"])

# %%
for this_file in config["paths"]["merged_file_list"]:
    print(this_file)
    full_output_df = pd.read_csv(this_file, dtype=str)
    print(full_output_df.shape)
    all_results(full_output_df, evaluation_cases_main)
