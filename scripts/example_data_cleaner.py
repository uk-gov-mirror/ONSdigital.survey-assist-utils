
from survey_assist_utils.configs.column_config import (
    ColumnConfig,
)
from survey_assist_utils.data_cleaning.data_cleaner import (
    DataCleaner,
)

import numpy as np
import pandas as pd
# Columns
coders_list = [f'sic_ind_occ{i}' for i in range(1, 4)]

config = ColumnConfig(
    model_label_cols=[], # not used in this test
    model_score_cols=[], # not used in this test
    clerical_label_cols=coders_list,
    id_col="unique_id",
)


df = pd.read_csv('data/evaluation_data/DSC_Rep_Sample.csv')

cleaner = DataCleaner(df, config)
clean_df = cleaner.process()

test = "KB056090"
print('Checking the addition of leading zeros in dataframe:')
print(clean_df.loc[clean_df['unique_id'] == test][coders_list])

# write the csv:
clean_df.to_csv("cleaned_data.csv", index = False)
