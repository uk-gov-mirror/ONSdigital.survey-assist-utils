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

from survey_assist_utils.evaluation.coder_alignment import (
    ColumnConfig,
    ConfusionMatrixConfig,
    LabelAccuracy,
    PlotConfig,
)


# --- Tests for LabelAccuracy ---
@pytest.fixture
def sample_data_and_config():  # pylint: disable=redefined-outer-name
    """A pytest fixture to create a standard set of test data and config."""
    test_data = pd.DataFrame(
        {
            "unique_id": ["A", "B", "C", "D", "E"],
            "clerical_label_1": ["12345", "1234", "-9", "nan", "5432x"],
            "clerical_label_2": ["23456", np.nan, "4+", "", "54321"],
            "model_label_1": ["12345", "01234", "99999", "54321", "54322"],
            "model_label_2": ["99999", "12300", "54322", "88888", "54322"],
            "model_score_1": [0.9, 0.8, 0.99, 0.7, 0.85],
            "model_score_2": [0.1, 0.7, 0.98, 0.6, 0.80],
        }
    )

    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    return test_data, config


def test_init_and_cleaning(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests that the class initializes and cleans data correctly."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    assert analyzer.df.loc[1, "clerical_label_1"] == "01234"
    assert analyzer.df.loc[2, "clerical_label_1"] == "-9"
    assert pd.isna(analyzer.df.loc[3, "clerical_label_1"])


def test_add_derived_columns(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests that the derived columns are created with correct values."""
    df, config = sample_data_and_config
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
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests the Jaccard similarity calculation."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    # Corrected expected calculation for Jaccard:
    # A: int=1, uni=3 -> 1/3
    # B: int=1, uni=3 -> 1/3
    # C: Nan - No valid results
    # D: int=0, uni=2 -> 0
    # E: int=0, uni=4 -> 0
    # Mean = (1/3 + 1/3  + 0 + 0) / 4 = 0.17
    assert analyzer.get_jaccard_similarity() == pytest.approx(0.17, abs=0.01)



def test_jaccard_similarity_for_single_row():
    """
    Tests the Jaccard similarity logic for specific single-row scenarios.
    This effectively tests the internal `get_jaccard_row` function.
    """
    # --- Scenario 1: Partial Overlap ---
    test_data_partial = pd.DataFrame({
        "id": ["A"],
        "clerical_1": ["11111"],
        "clerical_2": ["22222"],
        "model_1": ["11111"],
        "model_2": ["33333"],
        "model_score_1": [0.9],
        "model_score_2": [0.1],
    })
    config = ColumnConfig(
        model_label_cols=["model_1", "model_2"],
        clerical_label_cols=["clerical_1", "clerical_2"],
        model_score_cols=["model_score_1", "model_score_2"], # Add a dummy score column
        id_col="id"
    )
    print('test_data_partial', test_data_partial)

    analyzer_partial = LabelAccuracy(df=test_data_partial, column_config=config)
    # Expected: Intersection={11111} (size 1), Union={11111, 22222, 33333} (size 3) -> 1/3
    assert analyzer_partial.get_jaccard_similarity() == pytest.approx(1/3)


    # --- Scenario 2: No Overlap ---
    test_data_none = pd.DataFrame({
        "id": ["B"],
        "clerical_1": ["11111"],
        "model_1": ["22222"]
    })
    test_data_none["model_score_1"] = [1.0]
    config_none = ColumnConfig(
        model_label_cols=["model_1"],
        clerical_label_cols=["clerical_1"],
        model_score_cols=["model_score_1"], # Add a dummy score column
        id_col="id"
    )

    analyzer_none = LabelAccuracy(df=test_data_none, column_config=config_none)
    # Expected: Intersection=0, Union=2 -> 0/2 = 0
    assert analyzer_none.get_jaccard_similarity() == 0.0


def test_get_candidate_contribution(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests the candidate contribution method for a single candidate."""
    df, config = sample_data_and_config
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


def test_validate_inputs_raises_errors(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests that __init__ raises ValueErrors for bad configuration."""
    df, _ = sample_data_and_config

    # Test for missing column
    bad_config_missing = ColumnConfig(
        model_label_cols=["model_label_1", "missing_col"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="missing_id",
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        LabelAccuracy(df=df, column_config=bad_config_missing)

    # Test for mismatched score/label columns
    bad_config_mismatch = ColumnConfig(
        model_label_cols=["model_label_1"],
        model_score_cols=["model_score_1", "model_score_2"],
        clerical_label_cols=["clerical_label_1", "clerical_label_2"],
        id_col="unique_id",
    )
    with pytest.raises(ValueError, match="must match number of score columns"):
        LabelAccuracy(df=df, column_config=bad_config_mismatch)


def test_get_accuracy_thoroughly(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests the get_accuracy method with more scenarios."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test 2-digit match accuracy (3 correct out of 5 = 60%)
    result = analyzer.get_accuracy(match_type="2-digit", extended=True)
    assert result["accuracy_percent"] == pytest.approx(60.0, abs=0.01)

    # Test with a threshold that filters the data
    # At threshold 0.8, rows A, B, C, E remain.
    # is_correct: A=True, B=True, C=False, E=False. -> 2/4 = 50%
    result_thresh = analyzer.get_accuracy(threshold=0.8, extended=True)

    assert result_thresh["accuracy_percent"] == pytest.approx(50, abs=0.01)
    assert result_thresh["total_considered"] == 4  # noqa: PLR2004

    # Test edge case where no data meets threshold
    assert analyzer.get_accuracy(threshold=1.0) == pytest.approx(0.0, abs=0.01)


def test_get_coverage(sample_data_and_config):  # pylint: disable=redefined-outer-name
    """Tests the get_coverage method."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # 4 of 5 scores are >= 0.8, so coverage is 80%
    assert analyzer.get_coverage(threshold=0.8) == pytest.approx(80, abs=0.01)
    # All scores are >= 0.1, so coverage is 100%
    assert analyzer.get_coverage(threshold=0.1) == pytest.approx(100, abs=0.01)


def test_get_summary_stats(
    sample_data_and_config,
):  # pylint: disable=redefined-outer-name
    """Tests that get_summary_stats returns a dictionary with correct keys and values."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)
    stats = analyzer.get_summary_stats()

    assert isinstance(stats, dict)
    assert stats["total_samples"] == 5  # noqa: PLR2004
    # Overall accuracy (full match) is 2/5 = 40%
    assert stats["overall_accuracy"] == pytest.approx(40, abs=0.01)


def test_plotting_functions_run_without_error(
    sample_data_and_config: tuple, monkeypatch, tmp_path
):  # pylint: disable=redefined-outer-name
    """Tests that plotting functions run without raising an error."""
    monkeypatch.setattr(plt, "show", lambda: None)
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    # Test plotting with saving enabled
    plot_conf = PlotConfig(save=True, filename=str(tmp_path / "test_plot.png"))

    # Test threshold curve plot
    analyzer.plot_threshold_curves(plot_config=plot_conf)

    # Test confusion matrix plot
    matrix_conf = ConfusionMatrixConfig(
        human_code_col="clerical_label_1", llm_code_col="model_label_1"
    )
    analyzer.plot_confusion_heatmap(matrix_config=matrix_conf, plot_config=plot_conf)
    assert plot_conf.filename is not None, "Filename is None"
    assert os.path.isfile(plot_conf.filename), f"File not found: {plot_conf.filename}"


def test_save_output(
    sample_data_and_config, tmp_path
):  # pylint: disable=redefined-outer-name
    """Tests that save_output correctly creates files in a temporary directory."""
    df, config = sample_data_and_config
    analyzer = LabelAccuracy(df=df, column_config=config)

    metadata = {"evaluation_type": "test_run", "model_version": "1.0"}
    eval_result = {"accuracy": 100.0}

    # Call the function, directing output to the temporary path provided by pytest
    output_folder = analyzer.save_output(
        metadata=metadata, eval_result=eval_result, save_path=str(tmp_path)
    )

    # Check that the folder and files were created
    assert os.path.isdir(output_folder)
    metadata_path = os.path.join(output_folder, "metadata.json")
    eval_path = os.path.join(output_folder, "evaluation_result.json")
    assert os.path.isfile(metadata_path)
    assert os.path.isfile(eval_path)

    # Check the content of one of the files
    with open(metadata_path, encoding="utf-8") as f:
        saved_meta = json.load(f)
    assert saved_meta["model_version"] == "1.0"

    # Test that it raises an error for empty metadata
    with pytest.raises(ValueError, match="Metadata dictionary cannot be empty"):
        analyzer.save_output(metadata={}, eval_result=eval_result)
