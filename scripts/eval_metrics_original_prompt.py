# pylint: disable=R0801, W0511
import re
from collections import namedtuple

import pandas as pd

from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy


def parse_sic_candidates(candidates_str):
    """Parse SIC candidates from SicCandidate string format."""
    if (
        pd.isna(candidates_str)
        or candidates_str == ""
        or str(candidates_str).lower() == "nan"
    ):
        return []

    # Create named tuple type
    SicCandidate = namedtuple(
        "SicCandidate", ["sic_code", "sic_descriptive", "likelihood"]
    )

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"SicCandidate\(sic_code='([^']+)', sic_descriptive='([^']+)', likelihood=([0-9.]+)\)"  # pylint: disable=C0301
        matches = re.findall(pattern, str(candidates_str))

        candidates = []
        for match in matches:
            candidate = SicCandidate(
                sic_code=match[0], sic_descriptive=match[1], likelihood=float(match[2])
            )
            candidates.append(candidate)

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
    except Exception:  # pylint: disable=W0706 # TODO: introduce logging
        raise


# 1. Load your prepared data
# my_dataframe = pd.read_pickle("../data/one_prompt_rand100_stg2_2025_08_11_13.gz")
my_dataframe = pd.read_csv("../data/one_prompt_rand100_stg2_2025_08_11_13.csv")

# 2. Get clerically coded data
cc_data = pd.read_csv("../data/tlfs_2k_eval_set.csv")

# 3. Merge datasets
merged_data = my_dataframe.merge(
    cc_data[
        [
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
        ]
    ],
    left_on="unique_id",
    right_on="unique_id",
    how="left",
)

# 4. Parse sic_candidates from LLM output
merged_data["sic_candidate_list"] = merged_data["sic_candidates"].apply(
    parse_sic_candidates
)

# 5. Unpack sic_candidates into new individual columns
for i in range(0, 10):
    merged_data["sic_candidate_" + str(i + 1)] = merged_data[
        "sic_candidate_list"
    ].apply(lambda x, i=i: x[i].sic_code if len(x) > i else "")

for i in range(0, 10):
    merged_data["sic_candidate_score_" + str(i + 1)] = 0.5

# 5. Unpack CC labels
merged_data["cc_list"] = merged_data["All_Clerical_codes"].apply(parse_clerical_code)

for i in range(0, 5):
    merged_data["cc_candidate_" + str(i + 1)] = merged_data["cc_list"].apply(
        lambda x, i=i: x[i] if len(x) > i else ""
    )

# 7. Define the configuration for the test
# This example compares the top 3 model suggestions against 2 human-coded columns
col_config = ColumnConfig(
    model_label_cols=["sic_candidate_1", "sic_candidate_2"],
    model_score_cols=["sic_candidate_score_1", "sic_candidate_score_2"],
    clerical_label_cols=["cc_candidate_1", "cc_candidate_2"],
    id_col="unique_id",
    filter_unambiguous=True,  # Only analyze unambiguous records in this case
)

# 3. Initialise the analyser with the DataFrame and config
analyzer = LabelAccuracy(df=my_dataframe, column_config=col_config)

# 4. Run the desired analysis
# Get detailed accuracy stats for full 5-digit matches
accuracy_stats = analyzer.get_accuracy(
    match_type="2-dig", extended=True
)  # match_type can be "2-digit"

# Calculate the average suggestion quality
# jaccard_score = analyzer.get_jaccard_similarity()

# Generate a confusion matrix heatmap
# analyzer.plot_confusion_heatmap(
#     human_code_col="CC_1",
#     llm_code_col="SA_1"
# )

print(f"Accuracy Stats: {accuracy_stats}")
# print(f"Average Jaccard Score: {jaccard_score:.4f}")
