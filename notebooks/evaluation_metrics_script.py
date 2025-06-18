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
import json
import logging
import os
from datetime import datetime
import importlib
import toml

from collections import defaultdict
from dataclasses import dataclass

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import survey_assist_utils.evaluation.coder_alignment as coder_alignment
importlib.reload(coder_alignment)

from survey_assist_utils.evaluation.coder_alignment import LabelAccuracy

# %%
# Read in the merged data file:
merged_df =  pd.read_csv('/home/user/survey-assist-utils/data/evaluation_data/combined_outputs.csv', dtype=str)
print('merged_df', merged_df.shape)

# %%
# Set up some lists to hold the columns we are interested in:
candidate_list = ['candidate_1_sic_code', 'candidate_2_sic_code',
       'candidate_3_sic_code', 'candidate_4_sic_code',
       'candidate_5_sic_code']
candidate_likelihood = ['candidate_1_likelihood', 'candidate_2_likelihood',
       'candidate_3_likelihood', 'candidate_4_likelihood',
       'candidate_5_likelihood']
coders_list = ["sic_ind_occ1",
        "sic_ind_occ2",
        "sic_ind_occ3"]

# %%
# Create the object to work on with the analysis:
analyzer = LabelAccuracy(
    merged_df,
    id_col="unique_id",
    desc_col="unique_id",
    model_label_cols=candidate_list,
    model_score_cols=candidate_likelihood,
    clerical_label_cols=coders_list
)

# %%
# Basic accuracy
digit_list = ['2-digit', 'full']
type_digits = digit_list[0]
print(f"Overall accuracy {type_digits},  {analyzer.get_accuracy(threshold = 0.0, match_type = type_digits):.1f}%")
type_digits = digit_list[1]
print(f"Overall accuracy {type_digits},  {analyzer.get_accuracy(threshold = 0.0, match_type = type_digits):.1f}%")




# %%
analyzer.get_coverage()

# %%
analyzer.get_threshold_stats()

# %%
analyzer.plot_threshold_curves()

# %%
analyzer.get_summary_stats()

# %%
human_code_col = "sic_ind_occ1"
llm_code_col = 'chosen_sic_code'
exclude_patterns = ['x', '-9']
analyzer.plot_confusion_heatmap(
        human_code_col = human_code_col,
        llm_code_col = llm_code_col,
        top_n = 10,
        exclude_patterns = exclude_patterns)

# %%
# Confusion matrix:
merged_df =  pd.read_csv('/home/user/survey-assist-utils/data/evaluation_data/combined_outputs.csv', dtype=str)

print('merged_df', merged_df.shape)


candidate_list = ['chosen_sic_code', 'candidate_2_sic_code',
       'candidate_3_sic_code', 'candidate_4_sic_code',
       'candidate_5_sic_code']
candidate_likelihood = ['candidate_1_likelihood', 'candidate_2_likelihood',
       'candidate_3_likelihood', 'candidate_4_likelihood',
       'candidate_5_likelihood']
coders_list = ["sic_ind_occ1",
        "sic_ind_occ2",
        "sic_ind_occ3"]

CC_first_choice = coders_list[0]
human_code_col = CC_first_choice      # The ground truth from the human coder
llm_code_col = candidate_list[0]    


columns_to_keep = [human_code_col, llm_code_col, 'unique_id']# your list of columns
print(columns_to_keep)
n = len(merged_df)
# Create the new DataFrame
new_df = merged_df.loc[:n, columns_to_keep]
print('new_df', new_df.shape)

# Define how many "top" codes you want to see. Let's start with the top 5.
N = 12
# Get the most frequent codes from the human coder column
new_df = new_df[~new_df[human_code_col].str.contains('x')]
new_df = new_df[~new_df[human_code_col].str.contains('-9')]
top_human_codes = new_df[human_code_col].value_counts().nlargest(N).index
print(f"Top {N} Human Codes:\n{top_human_codes.tolist()}\n")

# Get the most frequent codes from the LLM's prediction column
top_llm_codes = merged_df[llm_code_col].value_counts().nlargest(N).index
print(f"Top {N} LLM Codes:\n{top_llm_codes.tolist()}\n")

# Create a new DataFrame that ONLY includes rows where the codes are in our top lists.
filtered_df = new_df[
    new_df[human_code_col].isin(top_human_codes) & 
    new_df[llm_code_col].isin(top_llm_codes)
]


# --- Step 3: Create the Confusion Matrix ---
# A confusion matrix is just a frequency table. pandas.crosstab is perfect for this.
# It counts how many times each pair of (Human, LLM) codes appears.
confusion_matrix = pd.crosstab(
    filtered_df[human_code_col],
    filtered_df[llm_code_col]
)

print("--- Confusion Matrix ---")
print(confusion_matrix)


# --- Step 4: Visualize as a Heatmap ---
# A heatmap makes the numbers in the matrix easy to interpret visually.
plt.figure(figsize=(10, 8)) # Set the figure size to make it readable

heatmap = sns.heatmap(
    confusion_matrix, 
    annot=True,     # This writes the numbers inside the squares
    fmt='d',        # Format the numbers as integers
    cmap='YlGnBu'   # A nice color scheme: Yellow -> Green -> Blue
)

plt.title(f'Confusion Matrix: Top {N} Human vs. LLM Codes', fontsize=16)
plt.ylabel('Human Coder (Ground Truth)', fontsize=12)
plt.xlabel('LLM Prediction', fontsize=12)
plt.tight_layout() # Adjust layout to make sure everything fits
plt.show()

