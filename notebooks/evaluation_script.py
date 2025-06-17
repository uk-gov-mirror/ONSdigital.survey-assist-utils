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

import pandas as pd
import numpy as np
import toml
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from sklearn.metrics import confusion_matrix
from typing import Tuple


from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor

import survey_assist_utils.evaluation.coder_alignment as coder_alignment
importlib.reload(coder_alignment)

from survey_assist_utils.evaluation.coder_alignment import AlignmentEvaluator, LabelAccuracy


# %%

# %%
CONFIG_PATH = "../config.toml"
preprocessor = JsonPreprocessor(config_path=CONFIG_PATH)
preprocessor.config

# %%
count = preprocessor.count_all_records()
print(count)
combined = preprocessor.process_files()
print(combined.shape)


merged_df2 = preprocessor.merge_eval_data(combined)
merged_df2.shape

# %%
merged_df2.to_csv('merged_df2.csv')

# %%
analyzer_alignment = AlignmentEvaluator(filepath='merged_df2.csv')


# %%
merged_df2.columns

# %%
candidate_list = ['candidate_1_sic_code', 'candidate_2_sic_code',
       'candidate_3_sic_code', 'candidate_4_sic_code',
       'candidate_5_sic_code']
candidate_likelihood = ['candidate_1_likelihood', 'candidate_2_likelihood',
       'candidate_3_likelihood', 'candidate_4_likelihood',
       'candidate_5_likelihood']
coders_list = ["sic_ind1",
        "sic_ind2",
        "sic_ind3"]
n = 1
m = 1        

# %%
# Assess the accuracy of number of digits
for digits in range(2, 6):
    print(digits)
    result = analyzer_alignment.calculate_match_rate(coders_list[0], candidate_list[0], n = digits)
    #result = analyzer_alignment.calculate_match_rate("sic_ind1", 'candidate_1_sic_code', n = digits)
    print(result)

# %%
import pandas as pd
import io

csv_data = """unique_id,CC_1,CC_2,CC_3,SA_1,SA_2,SA_3,SA_4,SA_5,SA_score_1,SA_score_2,SA_score_3,SA_score_4,SA_score_5
test_001,11111,22222,33333,44444,55555,11199,66666,77777,0.9,0.8,0.7,0.6,0.5
test_002,10001,20002,30003,40004,50005,20002,60006,30099,0.8,0.7,0.9,0.6,0.5
test_003,12345,67890,54321,54321,99999,88888,12345,77777,0.9,0.6,0.5,0.8,0.4
test_004,77700,88800,99900,11100,88800,99900,22200,77700,0.5,0.9,0.8,0.4,0.7
test_005,25000,,75000,25000,35000,,45000,75099,0.9,0.8,0.7,0.6,0.5
"""

# Define the data types to ensure 5-digit codes are treated as text (strings)
string_cols = {
    'CC_1': str, 'CC_2': str, 'CC_3': str,
    'SA_1': str, 'SA_2': str, 'SA_3': str, 'SA_4': str, 'SA_5': str
}

# Read the data into a pandas DataFrame
df = pd.read_csv(io.StringIO(csv_data), dtype=string_cols)

print("Test data loaded successfully!")
print(df)
print("\nData types for CC columns:")
print(df[['CC_1', 'CC_2', 'CC_3']].info())

# %%
analyzer_alignment = AlignmentEvaluator(filepath='unit_test_digits_accuracy.csv')
coders_list = ['CC_1', 'CC_2', 'CC_3']
candidate_list = ['SA_1', 'SA_2', 'SA_3', 'SA_4', 'SA_5']
candidate_likelihood = ['SA_score_1', 'SA_score_2', 'SA_score_3', 'SA_score_4', 'SA_score_5']
for coders in coders_list:
    for candididate in candidate_list:
        # Assess the accuracy of number of digits
        for digits in range(2, 6):
            result = analyzer_alignment.calculate_match_rate(coders, candididate, n = digits)
            print(coders, candididate, digits, result)

# %%

# Define the data types to ensure 5-digit codes are treated as text (strings)
string_cols = {
    'CC_1': str, 'CC_2': str, 'CC_3': str,
    'SA_1': str, 'SA_2': str, 'SA_3': str, 'SA_4': str, 'SA_5': str
}
df = pd.read_csv('unit_test_label_accuracy.csv', dtype=string_cols)
for n in range(1,6):
    print(n, candidate_list[0:n])
    for m in range(1,4):
        print(m, coders_list[0:m])
        # Calculate overall classifAI accuracy at 5-digit level
        analyzer = LabelAccuracy(
            df,
            id_col="unique_id",
            desc_col="unique_id",
            model_label_cols=candidate_list[0:n],
            model_score_cols=candidate_likelihood[0:n],
            clerical_label_cols=coders_list[0:m]
        )
        print(f"Overall accuracy: {analyzer.get_accuracy():.1f}%")    

# %%

analyzer.get_accuracy()

# %%
analyzer.get_coverage()

# %%
analyzer.get_threshold_stats()

# %%
analyzer.plot_threshold_curves()

# %%
analyzer.get_summary_stats()

# %%

metadata_all = {
    "evaluation_type": "test",
    "coverage": "all_samples",
    "classification_type": "sic",
    "classifai_method": "NA",
    "knowledge_base": "NA",
}

stats = analyzer.get_summary_stats()
LabelAccuracy.save_output(metadata_all, stats)
