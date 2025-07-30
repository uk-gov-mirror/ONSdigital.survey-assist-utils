"""This module contains pytest configuration, hooks and shared fixtures.

It sets up a global logger and defines hooks for pytest to log events
such as the start and finish of a test session.

Functions:
    pytest_configure(config): Applies global test configuration.
    pytest_sessionstart(session): Logs the start of a test session.
    pytest_sessionfinish(session, exitstatus): Logs the end of a test session.
This module contains pytest configuration

Fixtures defined in this file are automatically discovered by pytest and can be
used in any test file within this directory without needing to be imported.

Fixtures:
    raw_data_and_config: Provides a consistent, raw DataFrame and ColumnConfig
                         object for use across multiple test suites.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# REFACTOR: Import the config class to be used in the shared fixture.
from survey_assist_utils.configs.column_config import ColumnConfig

# Add src directory to Python path
SRC_PATH = str(Path(__file__).parent.parent / "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

# Configure a global logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,  # Adjust level as needed (DEBUG, INFO, WARNING, ERROR, CRITICAL)
)


def pytest_configure(config):  # pylint: disable=unused-argument
    """Hook function for pytest that is called after command line options have been parsed
    and all plugins and initial configuration are set up.

    This function is typically used to perform global test configuration or setup
    tasks before any tests are executed.

    Args:
        config (pytest.Config): The pytest configuration object containing command-line
            options and plugin configurations.
    """
    logger.info("=== Global Test Configuration Applied ===")


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):  # pylint: disable=unused-argument
    """Pytest hook implementation that is executed at the start of a test session.

    This function logs a message indicating that the test session has started.

    Args:
        session: The pytest session object (not used in this implementation).
    """
    logger.info("=== Test Session Started ===")


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):  # pylint: disable=unused-argument
    """Hook function called after the test session ends.

    This function is executed after all tests have been run and the test session
    is about to finish. It can be used to perform cleanup or logging tasks.

    Args:
        session (Session): The pytest session object containing information
            about the test session.
        exitstatus (int): The exit status code of the test session. This indicates
            whether the tests passed, failed, or were interrupted.

    Note:
        The `pylint: disable=unused-argument` directive is used to suppress
        warnings for unused arguments in this function.
    """
    logger.info("=== Test Session Finished ===")


# REFACTOR: The sample_data_and_config fixture has been moved here from the
# test_coder_alignment.py file so it can be shared across all test files.
# It is now renamed to 'raw_data_and_config' for clarity.
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
