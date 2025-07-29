"""column_config.py.

This module defines the ColumnConfig data class, which serves as a centralised
configuration object for specifying column names used in data processing workflows.

The configuration includes identifiers for model-generated labels and scores,
clerical labels, and optional filtering behavior. It is designed to be passed
into data cleaning or evaluation components to ensure consistent column references
across the pipeline.

Usage:
    config = ColumnConfig(
        model_label_cols=["model_label_1", "model_label_2"],
        model_score_cols=["score_1", "score_2"],
        clerical_label_cols=["clerical_label"],
        id_col="record_id",
        filter_unambiguous=True
    )
"""

from dataclasses import dataclass


# pylint: disable=too-few-public-methods
@dataclass
class ColumnConfig:
    """A data structure to hold the name configurations for the analysis."""

    model_label_cols: list[str]
    model_score_cols: list[str]
    clerical_label_cols: list[str]
    id_col: str = "id"
    filter_unambiguous: bool = False
