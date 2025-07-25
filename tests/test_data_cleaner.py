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


@pytest.fixture
def raw_data_and_config() -> tuple[pd.DataFrame, ColumnConfig]:
    """A pytest fixture to create a standard set of RAW test data and config."""
    test_data = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
            "Unambiguous": [True, True, False, True, True],
        }
    )
    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config


# --- Tests for DataCleaner ---
class TestDataCleaner:
    """Tests the DataCleaner's ability to validate and clean the DataFrame."""

    def test_cleaning_process(
        self, raw_data_and_config: tuple
    ):  # pylint: disable=redefined-outer-name
        """Tests that the main process method correctly cleans data."""
        df, config = raw_data_and_config
        cleaner = DataCleaner(df, config)
        clean_df = cleaner.process()

        # Check zfill padding
        assert clean_df.loc[1, "clerical_label_1"] == "01234"
        # Check special value preservation
        assert clean_df.loc[2, "clerical_label_1"] == "-9"
        # Check that string 'nan' and empty strings become real NaNs
        assert pd.isna(clean_df.loc[3, "clerical_label_1"])
        assert pd.isna(clean_df.loc[3, "clerical_label_2"])

    def test_unambiguous_filter(
        self, raw_data_and_config: tuple
    ):  # pylint: disable=redefined-outer-name
        """Tests that the unambiguous filter is applied correctly."""
        df, config = raw_data_and_config
        config.filter_unambiguous = True
        cleaner = DataCleaner(df, config)
        filtered_df = cleaner.process()
        # Row C has Unambiguous=False, so it should be removed.
        assert len(filtered_df) == 4  # noqa: PLR2004
        assert "C" not in filtered_df["unique_id"].values

    def test_validation_raises_errors(
        self, raw_data_and_config: tuple
    ):  # pylint: disable=redefined-outer-name
        """Tests that validation raises ValueErrors for bad configuration."""
        df, _ = raw_data_and_config
        bad_config = ColumnConfig(
            model_label_cols=["model_label_1", "missing_col"],
            model_score_cols=["model_score_1"],
            clerical_label_cols=["clerical_label_1"],
            id_col="unique_id",
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            DataCleaner(df, bad_config).process()


if __name__ == "__main__":
    unittest.main()
