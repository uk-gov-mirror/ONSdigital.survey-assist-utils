
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




Survey Assist SIC LLM 

Evaluation plan 

Updated – 14/07/2025 

[Author(s)] 

Overview 

Survey Assist (SA) is a developing prototype that aims to improve the codability and classification accuracy of Transformed Labour Force Survey (T-LFS) responses.  

Survey Assist can automatically generate the most relevant follow up survey questions in a live session with a respondent by making use of a Large Language Model (LLM). 

Evaluation covers a range of dimensions and starts with focus on alignment between clerical coder labels and LLM outputs. 

The purpose of the document is to provide a framework for evaluating the Survey Assist System. The LLM is prompted to assign a SIC code and follow-up question when given the survey respondents’ answers to questions about employment. 

SOC quality evaluation will be likely similar but is not covered here. 

The evaluation plan covers several aspects including: 

Alignment with clerical coders (CC). This refers to the extent to which outputs from Survey Assist agree with those provided by the clerical coders. 

Alignment can be evaluated both for coding without follow-up and again when additional information is collected from the respondents. 

Error analysis. An assessment will be carried out to understand the root cause of the main errors in SA responses. 

Model output stability. Here we aim to measure the extent to which Survey Assist results change when model parameters change. 

Quality of the follow-up. An assessment of reading score, suitability and ability to disambiguate the shortlist of possibilities. Exact details of this will be addressed elsewhere. 

Bias. A simulation of a survey using another LLM to play the part of the respondent. See bias document. 

 

 

 

 

Metrics of Alignment: 

 

 

 

 

 

 

Human coder alignment 

The dataset of 2000 broadly representative selection from across all sections. Provided by Clare Cheeseman and Hannah Tomlinson, these data are a carefully selected representative sample with expert SIC allocations that will be used to determine the benchmark of performance. 

Unambiguous responses. 

When a coder gives only a single assessment to five digits, they have identified unambiguously the code from the given information without need of a follow-up question. These will be used as a subset for measures. 

 

Scenario 1 – SA performance on unambiguously codable 

The purpose of this metric is to give an initial score of a direct match between the LLM’s first choice and the coders’ unambiguous choice. Of the initial 2079 dataset, 1291 are considered unambiguous. 

 

 

Business need/ motivation: 

This gives the team a baseline of SA’s ability to match the coders’ assessment, when the coder is sure of the answer. 

Approach: 

A measure of the number of responses where the first choices are the same, given as a percentage. This is done at both all five digits and again for only the first two digits (SIC Division) for both data sets for the unambiguous data only. 

How the metric will be used: 

This will be used to give an initial measure of Survey Assist’s direct accuracy. It will represent a minimum performance level against which proposed changes must improve upon before acceptance. 

 

Scenario 2 – Any shortlist match rate to unambiguously codable 

 

Survey Assist provides multiple answers as a shortlist. This metric checks if any of the SA answers match the coders’ selection for the unambiguous data. 

Business need/ motivation: 

This gives the team an insight as to whether the LLM is at least in the appropriate area of judgement with the human coders. 

Approach: 

A count of the number of responses where one or more SA responses matched the coders’ answer. This is done at both five digits and again for only the first two digits for both data sets. 

How the metric will be used: 

This will be used to check for minimum LLM adherence to the prompt and ability to make use of the RAG provided dataset. This is a ‘canary’ test and should always give a high result. 

All responses. 

 

Scenario 3 – Top Clerical Coder response featuring in SA shortlist rate 

 

This metric makes use of the coders’ tendency to put the preferred option into the first choice. Whilst not always the case, the rate at which this code featuring the SA shortlist is a valuable metric. 

Business need/ motivation: 

This provides a more nuanced measure of guidance of SA to be considering similar options to the coder. It will provide a basis for the follow-up question. 

Approach: 

A count of the number of responses where one of the SA responses matches the coders’ first choice. This is done at both five digits and two digits for both data sets, both as unambiguous only and all data. 

How the metric will be used: 

This will be used to provide a basis for prompt improvement and follow-up question relevance. 

Scenario 4 – Clerical Coder Shortlist and SA shortlist containing at least one common element 

 

This metric ...the rate at which this code featuring the SA shortlist is a valuable metric. 

Business need/ motivation: 

This provides. 

A follow-up question will be required, so absolute accuracy isn’t as important at this stage. 

Approach: 

A count of the number of responses where one of the SA responses is found in the CC’s shortlist. This is done at both five digits and two digits. 

 

How the metric will be used: 

This will be used to provide a basis for prompt improvement and follow-up question relevance. 

Scenario 5 – Jaccard measure. 

 

 

This metric . 

Business need/ motivation: 

This provides . 

Approach: 

A This is done at both five digits and two digits for both data sets, both as unambiguous only and all data. 

How the metric will be used: 

This will be used to provide  

 

 

 

 

 

 

 

Scenario 6 – Error analysis. 

 

 

This metric ... 

Business need/ motivation: 

This provides. 

A follow-up question will be required, so absolute accuracy isn’t as important at this stage. 

Approach: 

A count  

 

How the metric will be used: 

This will be used to provide  

 

 

Scenario 7 – Model Stability  

The purpose of this metric is to check the consistency of the system’s performance as prompts and models change during development.   

 

Business need/ motivation: 

This provides. 

Approach: 

A subset of a broad selection of data will be used as a quick check of performance.  

 

How the metric will be used: 

This will be used to provide a Go/No Go for incremental changes to ensure continuous improvement of the system. 

 

 