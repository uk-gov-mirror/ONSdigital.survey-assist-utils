import pandas as pd
import numpy as np
import re
from collections import namedtuple
from argparse import ArgumentParser as AP
from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy



############################### MODIFY the parse_args!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
def parse_args():
    """Parses command line arguments for the script."""
    parser = AP()
    parser.add_argument(
        "prepared_data", type=str,
        help="relative path to the prepared parquet dataset"
    )
    parser.add_argument(
        "cc_data",
        type=str,
        help="relative path to the CSV file containing clerically coded data",
    )
    # parser.add_argument(
    #     "output_folder",
    #     type=str,
    #     help="relative path to the output folder location (will be created if it doesn't exist)",
    # )
    # parser.add_argument(
    #     "--output_shortname",
    #     "-n",
    #     type=str,
    #     default="STG1",
    #     help="output filename prefix for easy identification (optional, default: STG1)",
    # )
    # parser.add_argument(
    #     "--batch_size",
    #     "-b",
    #     type=int,
    #     default=200,
    #     help="save the output every X rows, as a checkpoint that can be used to restart the "
    #     "processing job if needed (optional, default: 200)",
    # )
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
    if (pd.isna(candidates_array)).all() or str(candidates_array) == "" or str(candidates_array).lower() == "nan":
        return []

    # Create named tuple type
    SicCandidate = namedtuple('SicCandidate', ['likelihood', 'sic_code'])
    

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"\{'likelihood':\s*[\d\.]+,\s*'sic_code':\s*'[\d]+'\}"

        candidates = []
        for item in candidates_array:
            likelihood = item['likelihood']
            sic_code = item['sic_code']
            candidates.append(SicCandidate(likelihood, sic_code))

        return candidates
    except Exception as e:
        return []


def parse_clerical_code(candidates_str: str):
    if pd.isna(candidates_str) or candidates_str == "" or str(candidates_str).lower() == "nan":
        return []

    try:
        # Extract all RagCandidate entries using regex
        pattern = r"([0-9]+x*X*)"
        matches = re.findall(pattern, str(candidates_str))

        return matches
    except Exception as e:
        raise


def get_prepared_data(ss_df_path):
    return pd.read_parquet(ss_df_path)

def get_cc_data(cc_df_path):
    return pd.read_csv(cc_df_path)

# 1. Load your prepared data
# my_dataframe = pd.read_parquet("data/stage_2_one_prompt_20.parquet")
# my_dataframe = pd.read_parquet("data/stage_2_one_prompt_2k.parquet")

# # 2. Get clerically coded data
# cc_data = pd.read_csv("data/tlfs_2k_eval_set.csv")

# # 3. Merge datasets
# merged_data = my_dataframe[['unique_id', 'sic_section', 'sic2007_employee', 'sic2007_self_employed',
#     'sic_ind1', 'sic_ind2', 'sic_ind3', 'sic_ind_code_flag',
#     'soc2020_job_title', 'soc2020_job_description', 'sic_ind_occ1',
#     'sic_ind_occ2', 'sic_ind_occ3', 'sic_ind_occ_flag',
#     'semantic_search_results', 'final_sic_code', 'sic_candidates']].merge(cc_data[['unique_id', 'Not_Codeable',
#     'Four_Or_More', 'SIC_Division', 'num_answers', 'All_Clerical_codes',
#     'Match_5_digits', 'Match_3_digits', 'Match_2_digits', 'Unambiguous']], left_on="unique_id", right_on="unique_id", how = "left")

# # 4. Parse sic_candidates from LLM output
# merged_data['sic_candidate_list'] = merged_data['sic_candidates'].apply(parse_sic_candidates)

# # 5. Unpack sic_candidates into new individual columns
# for i in range(0,10):
#     merged_data['sic_candidate_' + str(i+1)] = merged_data['sic_candidate_list'].apply(lambda x: x[i].sic_code if len(x)>i else '')

# for i in range(0,10):
#     merged_data['sic_candidate_score_' + str(i+1)] = 0.5

# # 6. Unpack CC labels
# merged_data['cc_list'] = merged_data['All_Clerical_codes'].apply(parse_clerical_code)

# for i in range(0,5):
#     merged_data['cc_candidate_' + str(i+1)] = merged_data['cc_list'].apply(lambda x: x[i] if len(x)>i else '')

# # 7. Define the configuration for the test
# # This example compares the top 3 model suggestions against 2 human-coded columns
# col_config = ColumnConfig(
#     model_label_cols=["sic_candidate_1", "sic_candidate_2"],
#     model_score_cols=["sic_candidate_score_1", "sic_candidate_score_2"],
#     clerical_label_cols=["cc_candidate_1", "cc_candidate_2"],
#     id_col="unique_id",
#     filter_unambiguous=True  # Only analyze unambiguous records in this case
# )

# # 3. Initialise the analyser with the DataFrame and config
# analyzer = LabelAccuracy(df=merged_data, column_config=col_config)

# # 4. Run the desired analysis
# # Get detailed accuracy stats for full 5-digit matches
# accuracy_stats = analyzer.get_accuracy(match_type="2-digit", extended=True) #match_type can be "2-digit"

# # Calculate the average suggestion quality
# jaccard_score = analyzer.get_jaccard_similarity()

# # Generate a confusion matrix heatmap
# # analyzer.plot_confusion_heatmap(
# #     human_code_col="CC_1",
# #     llm_code_col="SA_1"
# # )

# print(f"Accuracy Stats: {accuracy_stats}")
# print(f"Average Jaccard Score: {jaccard_score:.4f}")



if __name__ == "__main__":
    args = parse_args()

    prepared_data = get_prepared_data(args.prepared_data)
    cc_data = get_cc_data(args.cc_data)
    print(prepared_data)