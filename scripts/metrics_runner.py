"""Evaluation Runner Script for Survey Assist.

This script loads a processed dataset and a TOML configuration file that defines
multiple evaluation scenarios. For each scenario, it calculates and prints
performance metrics such as full match accuracy, 2-digit match accuracy, and
Jaccard similarity.

Main Features:
--------------
- Loads evaluation cases from a TOML config file.
- Loads processed classification data from a CSV file.
- Dynamically configures evaluation columns based on the config.
- Computes and displays:
    - Full match accuracy
    - 2-digit match accuracy
    - Jaccard similarity index
    - Summary statistics

Usage:
------
Run from the command line with:
    python metrics_runner.py <processed_file.csv> <evaluation_config.toml>

Arguments:
----------
- processed_file: Path to the processed CSV file containing merged classification data.
- config_file: Path to the TOML file defining evaluation cases.

Dependencies:
-------------
- pandas
- numpy
- toml
- survey_assist_utils (internal package)

Example:
--------
    python metrics_runner.py data/final_output.csv configs/eval_config.toml
"""

import argparse
import logging
from collections.abc import Hashable
from typing import Any, Union

import numpy as np
import pandas as pd
import toml

from survey_assist_utils.configs.column_config import ColumnConfig
from survey_assist_utils.evaluation.coder_alignment import LabelAccuracy

PandasDtype = Union[
    pd.api.extensions.ExtensionDtype,
    str,
    np.dtype,
    type[str],
    type[complex],
    type[bool],
    type[object],
]


def load_config(config_path: str) -> list[dict[str, Any]]:
    """Load evaluation cases from a TOML config file."""
    try:
        config = toml.load(config_path)
        return config.get("evaluation_cases", [])
    except FileNotFoundError:
        print(f"Configuration file not found at {config_path}")
        return []


def load_data(data_path: str, string_columns: list[str]) -> pd.DataFrame:
    """Load processed data with specified string columns."""
    dtypes: dict[Hashable, PandasDtype] = dict.fromkeys(string_columns, "string")
    try:
        df = pd.read_csv(data_path, dtype=dtypes)
        print(f"Successfully loaded data. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Processed data file not found at {data_path}")
        return pd.DataFrame()


def print_accuracy(label: str, result: float | dict[str, Any]) -> None:
    """Print formatted accuracy results."""
    if isinstance(result, dict):
        print(
            f"{label}: {result['accuracy_percent']:.2f}% "
            f"({result['matches']}/{result['total_considered']})"
        )
    else:
        print(f"{label}: {result:.2f}")


def run_evaluation_case(case: dict[str, Any], df: pd.DataFrame, idx: int) -> None:
    """Run a single evaluation case."""
    title = case.get("Title", f"Unnamed Case {idx+1}")
    print(f"▶️  Running Case {idx+1}: {title}")

    num_llm = case.get("LLMs", [5])[0]
    num_cc = case.get("CCs", [3])[0]

    model_label_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
    model_score_cols = [f"candidate_{i}_likelihood" for i in range(1, 6)]
    clerical_label_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]

    case_config = ColumnConfig(
        model_label_cols=model_label_cols[:num_llm],
        model_score_cols=model_score_cols[:num_llm],
        clerical_label_cols=clerical_label_cols[:num_cc],
        id_col="unique_id",
        filter_unambiguous=case.get("Unambiguous", False),
    )

    analyzer = LabelAccuracy(df=df, column_config=case_config)

    full_match = analyzer.get_accuracy(match_type="full", extended=True)
    partial_match = analyzer.get_accuracy(match_type="2-digit", extended=True)

    print("\nFull Match Details:")
    print_accuracy("🎯 Full Match Accuracy", full_match)

    print("\n2-Digit Match Details:")
    print_accuracy("🔢 2-Digit Match Accuracy", partial_match)

    print("\n--- 📊 Overall Summary ---")
    summary = analyzer.get_summary_stats()
    jaccard = analyzer.get_jaccard_similarity()
    print(f"Total Records Analyzed:     {summary['total_samples']}")
    print(f"Jaccard Similarity Index:   {jaccard:.4f}")
    print(f"Overall Accuracy (Full):    {summary['overall_accuracy']:.2f}%")
    print("-" * 30)


def main(config_path: str, data_path: str) -> None:
    """Main function to run all evaluation cases."""
    print(f"Loading evaluation config from: {config_path}")
    evaluation_cases = load_config(config_path)

    model_cols = [f"candidate_{i}_sic_code" for i in range(1, 6)]
    clerical_cols = [f"sic_ind_occ{i}" for i in range(1, 4)]
    string_columns = model_cols + clerical_cols

    print(f"Loading processed data from: {data_path}")
    df = load_data(data_path, string_columns)

    if df.empty or not evaluation_cases:
        logging.warning("No data or evaluation cases to process.")
        return

    print("\n" + "=" * 70)
    for idx, case in enumerate(evaluation_cases):
        run_evaluation_case(case, df, idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation metrics.")
    parser.add_argument("processed_file", help="Path to processed CSV file")
    parser.add_argument("config_file", help="Path to TOML config file")
    args = parser.parse_args()

    main(args.config_file, args.processed_file)
