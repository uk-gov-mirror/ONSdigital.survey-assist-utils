"""test_column_config.py.

Unit tests for the ColumnConfig data class used in the survey-assist-utils package.

These tests verify that default values are correctly applied and that custom
values are properly assigned when the configuration object is instantiated.

Run this test module directly or include it in a larger test suite.
"""

import unittest

from survey_assist_utils.configs.column_config import ColumnConfig


class TestColumnConfig(unittest.TestCase):
    """Unit tests for the ColumnConfig data structure."""

    def test_defaults(self):
        """Test that default values are correctly set when optional arguments are omitted."""
        config = ColumnConfig(
            model_label_cols=["label1"],
            model_score_cols=["score1"],
            clerical_label_cols=["clerical1"],
        )
        self.assertEqual(config.id_col, "id")
        self.assertFalse(config.filter_unambiguous)

    def test_custom_values(self):
        """Test that custom values are correctly assigned to all fields."""
        config = ColumnConfig(
            model_label_cols=["label1", "label2"],
            model_score_cols=["score1", "score2"],
            clerical_label_cols=["clerical1"],
            id_col="custom_id",
            filter_unambiguous=True,
        )
        self.assertEqual(config.id_col, "custom_id")
        self.assertTrue(config.filter_unambiguous)


if __name__ == "__main__":
    unittest.main()
