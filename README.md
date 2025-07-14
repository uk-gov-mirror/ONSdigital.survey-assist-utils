# Survey Assist Utils

Utilities used as part of Survey Assist API or UI

## Overview

Survey Assist utility functions. These are common pieces of functionality that can be used by the UI or API.

## Features

- Generate a JWT token for authenticating to the API

## Prerequisites

Ensure you have the following installed on your local machine:

- [ ] Python 3.12 (Recommended: use `pyenv` to manage versions)
- [ ] `poetry` (for dependency management)
- [ ] Colima (if running locally with containers)
- [ ] Terraform (for infrastructure management)
- [ ] Google Cloud SDK (`gcloud`) with appropriate permissions

### Local Development Setup

The Makefile defines a set of commonly used commands and workflows.  Where possible use the files defined in the Makefile.

#### Clone the repository

```bash
git clone https://github.com/ONSdigital/survey-assist-utils.git
cd survey-assist-utils
```

#### Install Dependencies

```bash
poetry install
```

#### Run the Token Generator Locally

Set the following environment variables

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/GCP_CREDENTIALS.json"
export SA_EMAIL="GCP-SERVICE-ACCOUNT@SERVICE-ACCOUNT-ID.iam.gserviceaccount.com"
export JWT_SECRET=/path/to/GCP/secret.json
```

To generate an API token execute:

```bash
make generate-api-token
```

### GCP Setup

Placeholder

### Code Quality

Code quality and static analysis will be enforced using isort, black, ruff, mypy and pylint. Security checking will be enhanced by running bandit.

To check the code quality, but only report any errors without auto-fix run:

```bash
make check-python-nofix
```

To check the code quality and automatically fix errors where possible run:

```bash
make check-python
```

### Documentation

Documentation is available in the docs folder and can be viewed using mkdocs

```bash
make run-docs
```

### Testing

Pytest is used for testing alongside pytest-cov for coverage testing.  [/tests/conftest.py](/tests/conftest.py) defines config used by the tests.

Unit testing for utility functions is added to the [/tests/tests_utils.py](./tests/tests_utils.py)

```bash
make unit-tests
```

All tests can be run using

```bash
make all-tests
```

### Environment Variables

Placeholder

# Survey Assist SIC LLM - Evaluation Plan

## Overview

The purpose of this document is to provide a framework for evaluating the Survey Assist System. The LLM is prompted to assign a SIC code when given the survey respondents’ answers to questions about employment. We refer to this as Survey Assist (SA) in this document.

The LLM’s output consists of a set of ranked candidate SIC codes and their associated likelihood scores. This evaluation plan covers the primary metrics used to assess the alignment of SA's suggestions with those from expert human clerical coders (CC).

The analysis is performed by the `LabelAccuracy` class, which takes a DataFrame and a `ColumnConfig` object defining which columns to use for the evaluation.

## Human Coder Alignment

Primary dataset used to benchmark performance:

* **Dateset** A 2,000-record broadly representative sample from across all SIC sections, containing expert SIC assignments.

### Unambiguous Responses Subset

When a human coder provides only a single, complete 5-digit SIC code, the response is considered "Unambiguous." This subset is used for specific tests by setting the `filter_unambiguous=True` flag in the `ColumnConfig`, which provides a clean baseline for model performance.

## Core Evaluation Metrics

### Module coder_alignment.py

* by setting the @dataclass ColumnConfig we can set up alignment tests for any CC column to any SA column, for either Unambiguous, or all, allowing an N*M check of alignment.

### Metric 1: Match Accuracy

* **Purpose:** To measure how often any of the model's suggested codes match any of the codes provided by the human coder.

* **Business Need:** This is the primary measure of success. It tells us how often the model provides a correct answer within its suggestion list.

* **Approach:** The `get_accuracy()` method calculates this. It creates a boolean flag (`is_correct`) for each row, which is `True` if any code in the `model_label_cols` exists in the `clerical_label_cols`. The final metric is the percentage of `True` values. This can be calculated for both full 5-digit matches and partial 2-digit matches.

* **How it will be used:** This will be the main KPI for tracking overall model performance. It will also be used in conjunction with confidence scores (via the `get_threshold_stats()` method) to understand the trade-off between accuracy and automation coverage.

* This method is configurable to two digits or five digits of match. 

### Metric 2: Jaccard Similarity

* **Purpose:** To measure the quality and relevance of the entire suggestion list, not just the presence of a single correct answer.

* **Business Need:** A high accuracy score could still come from a noisy list (e.g., one correct code among four bad ones). The Jaccard score tells us how "clean" the suggestion list is. A high score means the model's suggestions closely mirror the human coder's choices.

* **Approach:** The `get_jaccard_similarity()` method calculates the average Jaccard Index (Intersection over Union) between the set of model codes and the set of clerical codes for each row.

* **How it will be used:** This metric provides a more nuanced view of performance. It will be used to evaluate if changes to prompts are making the suggestion lists more relevant and less noisy, which improves the user experience for the human coder.

### Metric 3: Candidate Contribution

* **Purpose:** To assess the "value add" of each individual candidate column (e.g., `candidate_3_sic_code`, `candidate_5_sic_code`).

* **Business Need:** This helps us decide if providing a long list of 5 candidates is useful or if a shorter list would be better. It answers the question: "How often does the 5th suggestion actually provide the correct answer?"

* **Approach:** The `get_candidate_contribution()` method is called for a specific candidate column. It calculates two things:

  1. How often that candidate matches the *primary* human code (`sic_ind_occ1`).

  2. How often that candidate matches *any* of the human codes.

* **How it will be used:** The results will inform decisions about the user interface (e.g., should we only show the top 3 candidates?) and guide efforts to improve the model's ranking ability.

### Metric 4: Confusion Matrix Heatmap

* **Purpose:** To visually identify systematic patterns of confusion between the model and human coders.

* **Business Need:** A single accuracy number doesn't tell us *what kind* of mistakes the model is making. A heatmap instantly reveals if the model consistently confuses two specific codes (e.g., wholesale vs. retail).

* **Approach:** The `plot_confusion_heatmap()` method creates a frequency table comparing the primary human code against the primary model code for the most common codes in the dataset, and displays it as a heatmap.

* **How it will be used:** This is a critical diagnostic tool. It will be used to identify specific "pitfalls" and provide concrete examples to guide prompt engineering and model fine-tuning.


## Other Analyses

* **Coder Type Differences:** The dataset can be segmented by the type of human coder (e.g., KB, CC, MC). The `LabelAccuracy` class can be instantiated for each subset to compare performance and identify any inconsistencies in human coding, if appropriated filtered.

* **Stability Testing:** This module can be used to provide feedback for whether changes in the system result in improvements or not by re-running against the dataset.

* **plot_threshold_curves** This plots the rate of confidence against coverage. It is an optinoal extra to assist in the evaluation.