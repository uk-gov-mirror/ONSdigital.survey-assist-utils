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
"""This script gives an example usage of coder_alignment.py."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

# %%
from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    ConfusionMatrixConfig,
    LabelAccuracy,
    PlotConfig,
)

# %% [markdown]
# # Example usage of coder_alignment.py

# %% [markdown]
# ### Load modules


# %% [markdown]
# ### Set up config variables

# %%
model_label_cols = [f"SA_{i}" for i in range(1, 6)]
model_score_cols = [f"SA_score_{i}" for i in range(1, 6)]
clerical_label_cols = [f"CC_{i}" for i in range(1, 4)]



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


# %% [markdown]
# ### Test All

# %%
# Note, this presumes that this file is run from survey-assist-utils/notebooks
relative_path = Path("e2e_test.csv")
df = pd.read_csv(relative_path, dtype=str)

analyzer_main = LabelAccuracy(df=df, column_config=config_main)

# do all tests:
full_acc_stats = analyzer_main.get_accuracy(match_type="full", extended=True)
print('full_acc_stats', full_acc_stats)
two_digit_acc_stats = analyzer_main.get_accuracy(match_type="2-digit", extended=True)
print('two_digit_acc_stats', two_digit_acc_stats)

jaccard_results = analyzer_main.get_jaccard_similarity()
print('jaccard_results', jaccard_results)
summary_stats = analyzer_main.get_summary_stats()

print('summary_stats', summary_stats)

thresholds = np.arange(0, 1.1, 0.1).tolist()
full_thresh_stats = analyzer_main.get_threshold_stats(thresholds = thresholds)
print(full_thresh_stats)

analyzer_main.plot_threshold_curves() 

# Configure to exclude the pattern 'x'
matrix_conf = ConfusionMatrixConfig(
    human_code_col=clerical_label_cols[0],
    llm_code_col=model_label_cols[0],
    exclude_patterns=['x']
)

analyzer_main.plot_confusion_heatmap(matrix_config=matrix_conf)

# %%
# Prepare meta date to check saving
example_meta_data = {'evaluation_type': 'example',
        'unit_tests':'PR example'}

eval_result = {
    "metadata": example_meta_data,
    "full_accuracy_stats": full_acc_stats.to_dict() if hasattr(full_acc_stats, "to_dict") else full_acc_stats,
    "two_digit_accuracy_stats": two_digit_acc_stats.to_dict() if hasattr(two_digit_acc_stats, "to_dict") else two_digit_acc_stats,
    "jaccard_similarity": jaccard_results.to_dict() if hasattr(jaccard_results, "to_dict") else jaccard_results,
    "summary_statistics": summary_stats.to_dict() if hasattr(summary_stats, "to_dict") else summary_stats,
    "threshold_statistics": full_thresh_stats.to_dict() if hasattr(full_thresh_stats, "to_dict") else full_thresh_stats
}

analyzer_main.save_output(
    metadata=example_meta_data,
    eval_result=eval_result, 
    save_path="data/"
)

# %%
analyzer_main.save_output(
    metadata=example_meta_data,
    eval_result=eval_result, 
    save_path="data/"
)

