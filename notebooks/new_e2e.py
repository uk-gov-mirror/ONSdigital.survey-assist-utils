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
# %run ../scripts/prepare_evaluation_data_for_analysis.py


# %%
from survey_assist_utils.evaluation.preprocessor import JsonPreprocessor
CONFIG_PATH = "config.toml"
preprocessor = JsonPreprocessor(config_path=CONFIG_PATH)
preprocessor.config

# %%
combined = preprocessor.process_files()

# %%
# Now merge in the SA output with the SIC test input:
merged_df2 = preprocessor.merge_eval_data(combined)
merged_df2.columns

# %%
# write the file locally:
merged_df2.to_csv('e2e_test.csv')
