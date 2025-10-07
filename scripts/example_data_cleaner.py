"""Script for cleaning and preparing evaluation data using the DataCleaner pipeline.

This script demonstrates how to use the `DataCleaner` class to validate, clean, and
standardize a raw evaluation DataFrame based on a configurable column schema. It is
intended for use in preprocessing steps prior to analysis or modeling.

Workflow:
    1. Define the column configuration using `ColumnConfig`, specifying which columns
       to validate and clean.
    2. Load the raw evaluation data from a CSV file.
    3. Initialize the `DataCleaner` with the column configuration.
    4. Run the `process()` method to apply validation, filtering, and cleaning.
    5. Optionally inspect specific records and export the cleaned DataFrame to disk.

Features:
    - Handles missing value formats and replaces them with `np.nan`.
    - Optionally filters out ambiguous records based on the 'Unambiguous' column.
    - Standardizes label columns (e.g., SIC codes) using safe zero-padding.
    - Preserves the original input data by operating on a copy.

Example:
    >>> config = ColumnConfig(
    ...     model_label_cols=[],
    ...     model_score_cols=[],
    ...     clerical_label_cols=["sic_ind_occ1", "sic_ind_occ2", "sic_ind_occ3"],
    ...     id_col="unique_id"
    ... )
    >>> df = pd.read_csv("data/evaluation_data/DSC_Rep_Sample.csv")
    >>> cleaner = DataCleaner(config)
    >>> clean_df = cleaner.process(df)
    >>> clean_df.to_csv("cleaned_data.csv", index=False)

Dependencies:
    - pandas
    - numpy
    - survey_assist_utils.configs.column_config
    - survey_assist_utils.data_cleaning.data_cleaner
"""

import pandas as pd

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.data_cleaning.data_cleaner import DataCleaner

# Define clerical label columns
coders_list = [f"sic_ind_occ{i}" for i in range(1, 4)]

# Create column configuration
config = ColumnConfig(
    model_label_cols=[],  # Not used in this test
    model_score_cols=[],  # Not used in this test
    clerical_label_cols=coders_list,
    id_col="unique_id",
)

# Load the raw evaluation data
df = pd.read_csv("data/evaluation_data/DSC_Rep_Sample.csv")

# Initialize the cleaner with the column configuration
cleaner = DataCleaner(config)

# Process the DataFrame
clean_df = cleaner.process(df)

# Check a specific record for zero-padding
TEST_ID = "KB056090"
print("Checking the addition of leading zeros in dataframe:")
print(clean_df.loc[clean_df["unique_id"] == TEST_ID, coders_list])

# Save the cleaned DataFrame
clean_df.to_csv("cleaned_data.csv", index=False)
