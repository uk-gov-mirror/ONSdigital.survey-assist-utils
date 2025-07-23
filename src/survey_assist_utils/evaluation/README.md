
# Coder Alignment Evaluation Module

## Overview

This module provides a comprehensive framework for evaluating the performance of Survey Assist's SIC code classifications against human-provided ground truth labels.

The core of the module is the `LabelAccuracy` class, which takes a DataFrame containing both model predictions and human-coded labels and calculates a suite of metrics to measure their alignment. It is designed to be flexible, allowing for comparisons between any combination of model and human-coded columns.

## Core Workflow

The typical workflow for using this module is as follows:

1. **Prepare the Data:** Load a DataFrame that contains the unique ID for each record, one or more columns with human-coded SIC labels, and one or more columns for each of the model's suggested SIC labels and their corresponding confidence scores.

2. **Configure the Analysis:** Create an instance of the `ColumnConfig` dataclass. This object tells the `LabelAccuracy` class which columns to use for the analysis (e.g., which columns contain model predictions vs. human labels) and whether to filter for specific subsets, like "Unambiguous" records.

3. **Instantiate the Evaluator:** Create an instance of the `LabelAccuracy` class, passing in the prepared DataFrame and the `ColumnConfig` object. The class will automatically handle internal data cleaning, validation, and the creation of derived columns needed for analysis.

4. **Calculate Metrics:** Call the various methods on the `LabelAccuracy` instance to calculate performance metrics and generate visualizations.

## Usage Example

```python
import pandas as pd
from survey_assist_utils.evaluation.coder_alignment import ColumnConfig, LabelAccuracy

# 1. Load your prepared data
my_dataframe = pd.read_csv("path/to/your/evaluation_data.csv")

# 2. Define the configuration for the test
# This example compares the top 3 model suggestions against 2 human-coded columns
col_config = ColumnConfig(
    model_label_cols=["SA_1", "SA_2", "SA_3"],
    model_score_cols=["SA_score_1", "SA_score_2", "SA_score_3"],
    clerical_label_cols=["CC_1", "CC_2"],
    id_col="unique_id",
    filter_unambiguous=True  # Only analyze unambiguous records in this case
)

# 3. Initialise the analyser with the DataFrame and config
analyzer = LabelAccuracy(df=my_dataframe, column_config=col_config)

# 4. Run the desired analysis
# Get detailed accuracy stats for full 5-digit matches
accuracy_stats = analyzer.get_accuracy(match_type="full", extended=True)

# Calculate the average suggestion quality
jaccard_score = analyzer.get_jaccard_similarity()

# Generate a confusion matrix heatmap
analyzer.plot_confusion_heatmap(
    human_code_col="CC_1",
    llm_code_col="SA_1"
)

print(f"Accuracy Stats: {accuracy_stats}")
print(f"Average Jaccard Score: {jaccard_score:.4f}")