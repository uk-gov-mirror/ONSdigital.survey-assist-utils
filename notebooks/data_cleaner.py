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
"""this reads in a csv of annotated data and zero pads it,
sorts out nans and saves it as a csv with strings instead of numbers.
"""

# %%

import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.data_cleaning.data_cleaner import DataCleaner

# %%
# Columns
candidate_list = [f"candidate_{i}_sic_code" for i in range(1, 6)]
candidate_likelihood = [f"candidate_{i}_likelihood" for i in range(1, 6)]
coders_list = [f"sic_ind_occ{i}" for i in range(1, 4)]


# %%
config = ColumnConfig(
    model_label_cols=[],
    model_score_cols=[],
    clerical_label_cols=coders_list,
    id_col="unique_id",
)

# %%
df = pd.read_csv("../data/evaluation_data/DSC_Rep_Sample.csv")

# %%
cleaner = DataCleaner(df, config)
clean_df = cleaner.process()

# %%
TEST = "KB056090"
print(clean_df.loc[clean_df["unique_id"] == TEST][coders_list])

# %%
# write the csv:
clean_df.to_csv("cleaned_data.csv", index=False)
