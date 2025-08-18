import re
from argparse import ArgumentParser as AP
from collections import namedtuple

import numpy as np
import pandas as pd

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy


############################### MODIFY the parse_args!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def parse_args():
    """Parses command line arguments for the script."""
    parser = AP()
    parser.add_argument(
        "prepared_data", type=str, help="relative path to the prepared parquet dataset"
    )
    parser.add_argument(
        "cc_data",
        type=str,
        help="relative path to the CSV file containing clerically coded data",
    )
    parser.add_argument(
        "--prepared_data_columns",
        "-pcol",
        type=list,
        default=[
            "unique_id",
            "sic_section",
            "sic2007_employee",
            "sic2007_self_employed",
            "sic_ind1",
            "sic_ind2",
            "sic_ind3",
            "sic_ind_code_flag",
            "soc2020_job_title",
            "soc2020_job_description",
            "sic_ind_occ1",
            "sic_ind_occ2",
            "sic_ind_occ3",
            "sic_ind_occ_flag",
            "semantic_search_results",
            "final_sic_code",
            "sic_candidates",
        ],
        help="""A list of all columns from the prepared data to be included for
        the evaluation (optional, default: list[str])""",
    )
    parser.add_argument(
        "--clerical_data_columns",
        "-ccol",
        type=list,
        default=[
            "unique_id",
            "Not_Codeable",
            "Four_Or_More",
            "SIC_Division",
            "num_answers",
            "All_Clerical_codes",
            "Match_5_digits",
            "Match_3_digits",
            "Match_2_digits",
            "Unambiguous",
        ],
        help="""A list of all columns from the clerically coded data to be
        included for the evaluation (optional, default: list[str])""",
    )
    parser.add_argument(
        "--test_configuration",
        "-t",
        type=str,
        default="MM",
        help="""test configuration; specify the type one-to-one, one-to-many,
        many-to-one, many-to-many , select `OO`, `OM`, `MO`, or `MM`. Defaults to `MM`""",
    )
    # parser.add_argument(
    #     "--restart",
    #     "-r",
    #     action="store_true",
    #     default=False,
    #     help="try to restart a processing job (optional flag)",
    # )
    return parser.parse_args()


def parse_sic_candidates(candidates_str):
    """Parse SIC candidates from SicCandidate string format"""
    candidates_array = np.array(candidates_str)
    if (
        (pd.isna(candidates_array)).all()
        or str(candidates_array) == ""
        or str(candidates_array).lower() == "nan"
    ):
        return []

    # Create named tuple type
    SicCandidate = namedtuple("SicCandidate", ["likelihood", "sic_code"])

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"\{'likelihood':\s*[\d\.]+,\s*'sic_code':\s*'[\d]+'\}"

        candidates = []
        for item in candidates_array:
            likelihood = item["likelihood"]
            sic_code = item["sic_code"]
            candidates.append(SicCandidate(likelihood, sic_code))

        return candidates
    except Exception:
        return []


def parse_clerical_code(candidates_str: str):
    if (
        pd.isna(candidates_str)
        or candidates_str == ""
        or str(candidates_str).lower() == "nan"
    ):
        return []

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"([0-9]+x*X*)"
        matches = re.findall(pattern, str(candidates_str))

        return matches
    except Exception:
        raise


# Get prepared data
def get_prepared_data(ss_df_path):
    return pd.read_parquet(ss_df_path)


# Get clerically coded data
def get_cc_data(cc_df_path):
    return pd.read_csv(cc_df_path)  # Double check if that is CSV or parquet


# Merge two datasets, preparing for the evaluation
def merge_datasets(
    prepared_data, cc_data, prepared_data_columns, clerical_data_columns
):
    return prepared_data[prepared_data_columns].merge(
        cc_data[clerical_data_columns],
        left_on="unique_id",
        right_on="unique_id",
        how="left",
    )


# Unpack sic_candidates into new individual columns
def sic_candidates_individual(merged_data):
    for i in range(0, 10):
        merged_data["sic_candidate_" + str(i + 1)] = merged_data[
            "sic_candidate_list"
        ].apply(lambda x: x[i].sic_code if len(x) > i else "")

    for i in range(0, 10):
        merged_data["sic_candidate_score_" + str(i + 1)] = 0.5


# Unpack CC labels
def unpack_cc(merged_data):
    merged_data["cc_list"] = merged_data["All_Clerical_codes"].apply(
        parse_clerical_code
    )

    for i in range(0, 5):
        merged_data["cc_candidate_" + str(i + 1)] = merged_data["cc_list"].apply(
            lambda x: x[i] if len(x) > i else ""
        )


def test_configuration(
    compare,
):  # Future: parse how many columns form LLM compare to how many columns prepared by clerically coded.
    # This example compares the top 3 model suggestions against 2 human-coded columns
    # should we allow comparing different sets, e.g. 3v3, 5v2 (currently we do 3v2)
    """Choose from OO, MM, OM, MO, where O stands for One, M stands for Many.
    First initial for clerically coded, second initial for LLM prepared data.

    Args:
        compare (str, optional): _description_. Defaults to "MM".

    Returns:
        _type_: _description_
    """
    if compare == "MM":
        col_config = ColumnConfig(
            clerical_label_cols=["cc_candidate_1", "cc_candidate_2"],
            model_label_cols=["sic_candidate_1", "sic_candidate_2", "sic_candidate_3"],
            model_score_cols=[
                "sic_candidate_score_1",
                "sic_candidate_score_2",
                "sic_candidate_score_3",
            ],
            id_col="unique_id",
            filter_unambiguous=True,  # Only analyze unambiguous records in this case
        )
    elif compare == "OM":
        col_config = ColumnConfig(
            clerical_label_cols=["cc_candidate_1"],
            model_label_cols=["sic_candidate_1", "sic_candidate_2", "sic_candidate_3"],
            model_score_cols=[
                "sic_candidate_score_1",
                "sic_candidate_score_2",
                "sic_candidate_score_3",
            ],
            id_col="unique_id",
            filter_unambiguous=True,  # Only analyze unambiguous records in this case
        )
    elif compare == "MO":
        col_config = ColumnConfig(
            clerical_label_cols=["cc_candidate_1", "cc_candidate_2"],
            model_label_cols=["sic_candidate_1", "sic_candidate_2"],
            model_score_cols=["sic_candidate_score_1", "sic_candidate_score_2"],
            id_col="unique_id",
            filter_unambiguous=True,  # Only analyze unambiguous records in this case
        )
    else:  # compare == "OO":
        col_config = ColumnConfig(
            clerical_label_cols=["cc_candidate_1"],
            model_label_cols=["sic_candidate_1"],
            model_score_cols=["sic_candidate_score_1"],
            id_col="unique_id",
            filter_unambiguous=True,  # Only analyze unambiguous records in this case
        )
    return col_config


# # Calculate the average suggestion quality
# jaccard_score = analyzer.get_jaccard_similarity()

# print(f"Average Jaccard Score: {jaccard_score:.4f}")


if __name__ == "__main__":
    args = parse_args()

    # Read the data
    prepared_data = get_prepared_data(args.prepared_data)
    cc_data = get_cc_data(args.cc_data)

    # Merge two datasets
    merged_data = merge_datasets(
        prepared_data, cc_data, args.prepared_data_columns, args.clerical_data_columns
    )

    # Parse sic_candidates from LLM output
    merged_data["sic_candidate_list"] = merged_data["sic_candidates"].apply(
        parse_sic_candidates
    )

    sic_candidates_individual(merged_data)
    unpack_cc(merged_data)
    col_config = test_configuration(args.test_configuration)
    # Initialise the analyser with the DataFrame and config
    analyzer = LabelAccuracy(df=merged_data, column_config=col_config)
    # Run the desired analysis
    # Get detailed accuracy stats for full 5-digit matches
    accuracy_stats = analyzer.get_accuracy(
        match_type="2-digit", extended=True
    )  # match_type can be "2-digit"
    print(f"Accuracy Stats: {accuracy_stats}")
