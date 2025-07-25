"""test_data_cleaner.py.

Unit tests for the DataCleaner class in the survey-assist-utils package.

These tests verify the correctness of the data cleaning pipeline, including:
- Input validation
- Filtering of ambiguous records
- Standardization of label formats
- Handling of missing values

The tests use a shared fixture to provide consistent raw data and configuration
for each test case.
"""

import unittest

import numpy as np
import pandas as pd
import pytest

from survey_assist_utils.configs.column_config import (
    ColumnConfig,
)
from survey_assist_utils.data_cleaning.data_cleaner import (
    DataCleaner,
)


class TestDataCleaner(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
            "Unambiguous": [True, True, False, True, True],
        })
        self.config = ColumnConfig(
            model_label_cols=["model_label_1", "model_label_2"],
            model_score_cols=["model_score_1", "model_score_2"],
            clerical_label_cols=["clerical_label_1", "clerical_label_2"],
            id_col="unique_id",
        )

    def test_cleaning_process(self):
        cleaner = DataCleaner(self.df, self.config)
        clean_df = cleaner.process()
        # Check zfill padding
        assert clean_df.loc[1, "clerical_label_1"] == "01234"
        # Check special value preservation
        assert clean_df.loc[2, "clerical_label_1"] == "-9"
        # Check that string 'nan' and empty strings become real NaNs
        assert pd.isna(clean_df.loc[3, "clerical_label_1"])
        assert pd.isna(clean_df.loc[3, "clerical_label_2"])

    def test_unambiguous_filter(
        self
    ):  # pylint: disable=redefined-outer-name
        """Tests that the unambiguous filter is applied correctly."""
        self.config.filter_unambiguous = True
        cleaner = DataCleaner(self.df, self.config)
        filtered_df = cleaner.process()
        # Row C has Unambiguous=False, so it should be removed.
        assert len(filtered_df) == 4  # noqa: PLR2004
        assert "C" not in filtered_df["unique_id"].values

    def test_validation_raises_errors(
        self
    ):  # pylint: disable=redefined-outer-name
        """Tests that validation raises ValueErrors for bad configuration."""
        bad_config = ColumnConfig(
            model_label_cols=["model_label_1", "missing_col"],
            model_score_cols=["model_score_1"],
            clerical_label_cols=["clerical_label_1"],
            id_col="unique_id",
        )
        self.config = bad_config
        with pytest.raises(ValueError, match="Missing required columns"):
            DataCleaner(self.df, self.config).process()


if __name__ == "__main__":
    unittest.main()
