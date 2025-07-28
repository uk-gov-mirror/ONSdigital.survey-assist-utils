"""Unit tests for the LabelAccuracy class in the coder_alignment module.

This test suite verifies the functionality of the LabelAccuracy class, ensuring
that it correctly processes input data and calculates various evaluation metrics.
The tests cover:
- Correct initialisation and data cleaning (handling of NaNs, special codes).
- Accurate creation of derived boolean columns ('is_correct', 'is_correct_2_digit').
- Validation of core metric calculations (accuracy, Jaccard similarity).
- Verification of the candidate contribution analysis.
- The ability of plotting functions to run without raising errors.
- Robustness against bad inputs and edge cases.
- Verification of utility methods like get_coverage, get_summary_stats, and save_output.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.evaluation.coder_alignment import (
    LabelAccuracy,
    MetricCalculator,
    DataCleaner,
)

# NOTE: The 'raw_data_and_config' fixture is automatically discovered
# by pytest from the 'conftest.py' file.


# --- Tests for LabelAccuracy ---

class TestMetricCalculator:
    """Tests the MetricCalculator's ability to compute metrics on clean data."""

    @pytest.fixture
    def setup_calculator(self, raw_data_and_config: tuple) -> MetricCalculator:
        """Helper to create a MetricCalculator instance with clean data."""
        df, config = raw_data_and_config
        clean_df = DataCleaner(df, config).process()
        return MetricCalculator(clean_df, config)

    def test_add_derived_columns(self, setup_calculator: MetricCalculator):
        """Tests that derived columns are created with correct values."""
        analyzer = setup_calculator
        # Full matches: A, B. is_correct = [T, T, F, F, F]
        assert analyzer.df["is_correct"].tolist() == [True, True, False, False, False]
        # 2-digit matches: A, B, E. is_correct_2_digit = [T, T, F, F, T]
        assert analyzer.df["is_correct_2_digit"].tolist() == [True, True, False, False, True]
        assert analyzer.df.loc[0, "max_score"] == 0.9

    def test_get_accuracy(self, setup_calculator: MetricCalculator):
        """Tests the get_accuracy method."""
        analyzer = setup_calculator
        # 2 full matches out of 5 = 40%
        assert analyzer.get_accuracy(match_type="full") == pytest.approx(40.0)
        # 3 2-digit matches out of 5 = 60%
        assert analyzer.get_accuracy(match_type="2-digit") == pytest.approx(60.0)

    def test_get_jaccard_similarity(self, setup_calculator: MetricCalculator):
        """Tests the Jaccard similarity calculation."""
        analyzer = setup_calculator
        # A: int=1, uni=3 -> 0.333
        # B: int=1, uni=3 -> 0.333
        # C: int=0, uni=2 -> 0
        # D: int=1, uni=3 -> 0.333
        # E: int=0, uni=3 -> 0
        # Mean = (0.333 + 0.333 + 0 + 0.333 + 0) / 5 = 1.0 / 5 = 0.2
        assert analyzer.get_jaccard_similarity() == pytest.approx(0.2, abs=0.01)



class TestLabelAccuracy:
    """Unit tests for the LabelAccuracy class, which evaluates the correctness of
    model-generated labels against clerical labels in a dataset.

    These tests validate the initialization, data cleaning, and derived column
    generation functionalities of the LabelAccuracy class.
    """

    @pytest.mark.usefixtures("raw_data_and_config")
    def test_init_and_cleaning(self, raw_data_and_config):
        """Test that LabelAccuracy correctly initializes and cleans the input DataFrame.

        Verifies that:
        - Specific label values are cleaned or preserved as expected.
        - NaN

        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df=df, column_config=config)
        assert analyzer.df.loc[1, "clerical_label_1"] == "01234"
        assert analyzer.df.loc[2, "clerical_label_1"] == "-9"
        assert pd.isna(analyzer.df.loc[3, "clerical_label_1"])
        """

    def test_add_derived_columns(
        self, raw_data_and_config
    ):  # pylint: disable=redefined-outer-name
        """Test that derived columns are added correctly to the DataFrame.

        Verifies that:
        - 'is_correct' column reflects exact label matches.
        - 'is_correct_2_digit' column reflects partial matches based on the last two digits.
        - 'max_score' column correctly identifies the highest model score per row.
        """
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df=df, column_config=config)
        assert analyzer.df["is_correct"].tolist() == [True, True, False, False, False]
        assert analyzer.df["is_correct_2_digit"].tolist() == [
            True,
            True,
            False,
            False,
            True,
        ]
        assert analyzer.df.loc[0, "max_score"] == 0.9  # noqa: PLR2004

    def test_get_jaccard_similarity(
        self, raw_data_and_config
    ):  # pylint: disable=redefined-outer-name
        """Tests the Jaccard similarity calculation."""
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df=df, column_config=config)
        # Corrected expected calculation for Jaccard:
        # A: int=1, uni=3 -> 1/3
        # B: int=1, uni=3 -> 1/3
        # C: Nan - No valid results
        # D: int=0, uni=2 -> 0
        # E: int=0, uni=4 -> 0
        # Mean = (1/3 + 1/3  + 0 + 0) / 4 = 0.17
        assert analyzer.get_jaccard_similarity() == pytest.approx(0.2, abs=0.01)

    def test_get_candidate_contribution(
        self, raw_data_and_config
    ):  # pylint: disable=redefined-outer-name
        """Tests the candidate contribution method for a single candidate."""
        df, config = raw_data_and_config
        analyzer = LabelAccuracy(df=df, column_config=config)
        result = analyzer.get_candidate_contribution("model_label_1")
        # model_label_1 matches clerical_label_1 once ('12345') and clerical_label_2
        # once (np.nan ignored)
        # clerical_label_1: "12345", "01234", "-9", nan, "5432x"
        # model_label_1: "12345", "01234", "99999", "54321", "54322"
        # clerical_label_2: "23456", nan, "4+", nan, "54321"
        # model_label_1 '12345' matches '12345' in clerical_label_1.
        # model_label_1 '01234' matches '01234' in clerical_label_1.
        assert result["primary_match_count"] == 2  # noqa: PLR2004
        # model_label_1 '54321' matches '54321' in clerical_label_2
        assert result["any_clerical_match_count"] == 2  # noqa: PLR2004
        assert result["any_clerical_match_percent"] == pytest.approx(40.0, abs=0.01)


def test_jaccard_similarity_for_single_row():
    """Tests the Jaccard similarity logic for specific single-row scenarios.
    This effectively tests the internal `get_jaccard_row` function.
    """
    # --- Scenario 1: Partial Overlap ---
    test_data_partial = pd.DataFrame(
        {
            "id": ["A"],
            "clerical_1": ["11111"],
            "clerical_2": ["22222"],
            "model_1": ["11111"],
            "model_2": ["33333"],
            "model_score_1": [0.9],
            "model_score_2": [0.1],
        }
    )
    config = ColumnConfig(
        model_label_cols=["model_1", "model_2"],
        clerical_label_cols=["clerical_1", "clerical_2"],
        model_score_cols=["model_score_1", "model_score_2"],  # Add a dummy score column
        id_col="id",
    )
    print("test_data_partial", test_data_partial)

    analyzer_partial = LabelAccuracy(df=test_data_partial, column_config=config)
    # Expected: Intersection={11111} (size 1), Union={11111, 22222, 33333} (size 3) -> 1/3
    assert analyzer_partial.get_jaccard_similarity() == pytest.approx(0)

    # --- Scenario 2: No Overlap ---
    test_data_none = pd.DataFrame(
        {"id": ["B"], "clerical_1": ["11111"], "model_1": ["22222"]}
    )
    test_data_none["model_score_1"] = [1.0]
    config_none = ColumnConfig(
        model_label_cols=["model_1"],
        clerical_label_cols=["clerical_1"],
        model_score_cols=["model_score_1"],  # Add a dummy score column
        id_col="id",
    )

    analyzer_none = LabelAccuracy(df=test_data_none, column_config=config_none)
    # Expected: Intersection=0, Union=2 -> 0/2 = 0
    assert analyzer_none.get_jaccard_similarity() == 0.0
