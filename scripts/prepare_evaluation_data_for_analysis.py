"""Load and Filter utilities.

The .toml configuration file should include:
- The path to the test data file.
- Column names mapping if they differ from defaults.

The utilities allow for the following:
1. Loads the configuration from the .toml file.
2. Loads the test data.
3. Adds data quality flag columns to the DataFrame.
"""

import logging
import os

import pandas as pd
import toml

# REFACTOR: Import the new, centralised FlagGenerator class.
from survey_assist_utils.processing.flag_generator import FlagGenerator

# --- Default Configuration Values (if not found in config) ---
DEFAULT_OUTPUT_DIR = "analysis_outputs"
DEFAULT_SIC_OCC1_COL = "sic_ind_occ1"
DEFAULT_SIC_OCC2_COL = "sic_ind_occ2"
DEFAULT_SIC_OCC3_COL = "sic_ind_occ3"

SPECIAL_SIC_NOT_CODEABLE = "-9"
SPECIAL_SIC_MULTIPLE_POSSIBLE = "4+"

# --- Constants for Data Quality ---
EXPECTED_SIC_LENGTH = 5
X_COUNT_FOR_MATCH_3 = 2
X_COUNT_FOR_MATCH_2 = 3


# Load the config:
def load_config(config_path):
    """Loads configuration settings from a .toml file.

    Args:
        config_path (str): The path to the .toml configuration file.

    Returns:
        dict: A dictionary containing the configuration settings.

    Example:
        config = load_config("config.toml")
        print(config["settings"]["api_gateway"])
    """
    with open(config_path, encoding="utf-8") as file:
        configuration = toml.load(file)
    return configuration


def _extract_sic_division(
    sic_occ1_series: pd.Series,
    not_codeable_flag: pd.Series,
    multiple_possible_flag: pd.Series,
) -> pd.Series:
    """Extracts the first two digits (division) from the sic_ind_occ1 series.

    Args:
        sic_occ1_series (pd.Series): The Series containing sic_ind_occ1 codes (as strings).
        not_codeable_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '-9'.
        multiple_possible_flag (pd.Series): Boolean Series indicating where sic_ind_occ1 is '4+'.

    Returns:
        pd.Series: A Series containing the first two digits as strings, or an
                   empty string if not applicable or for special codes.
    """
    # Default to empty string
    sic_division = pd.Series("", index=sic_occ1_series.index, dtype=str)

    # Condition for valid extraction:
    # Must start with at least two digits AND not be a special code.
    # Using .str.match() ensures we are dealing with strings.
    starts_with_two_digits = sic_occ1_series.str.match(
        r"^[0-9]{2}"
    )  # Matches if starts with 2+ digits

    # Rows eligible for extraction
    eligible_for_extraction = (
        starts_with_two_digits & ~not_codeable_flag & ~multiple_possible_flag
    )

    # Extract first two digits for eligible rows
    sic_division[eligible_for_extraction] = sic_occ1_series[
        eligible_for_extraction
    ].str[:2]

    return sic_division


def main():
    """Main function to run the data preparation pipeline."""
    # --- 1. Load Configuration and Set Up Logging ---
    config = load_config("config.toml")
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO").upper()
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- 2. Get File Paths from Config ---
    try:
        analysis_filepath = config["paths"]["batch_filepath"]
        analysis_csv = config["paths"]["analysis_csv"]
    except KeyError as e:
        logging.error("Missing required path in config.toml: %s", e)
        return

    # --- 3. Load Raw Data ---
    try:
        sic_dataframe = pd.read_csv(analysis_filepath, delimiter=",", dtype=str)
        logging.info("Successfully loaded raw data from: %s", analysis_filepath)
    except FileNotFoundError:
        logging.error("Raw data file not found at: %s", analysis_filepath)
        return

    # --- 4. Use FlagGenerator to Add Quality Flags ---
    # REFACTOR: Instantiate the new FlagGenerator class and call its process method.
    # This replaces the old, local add_data_quality_flags function.
    flag_generator = FlagGenerator()
    sic_dataframe_with_flags = flag_generator.add_flags(sic_dataframe)

    # Add SIC division (2 digits)

    # --- 5. Save the Enriched DataFrame ---
    output_dir = os.path.dirname(analysis_csv)
    os.makedirs(output_dir, exist_ok=True)
    sic_dataframe_with_flags.to_csv(analysis_csv, index=False)
    logging.info("Successfully saved data with flags to: %s", analysis_csv)

    print("\n--- Analysis Complete ---")
    print(f"Processed {len(sic_dataframe)} rows.")
    print(f"Output saved to {analysis_csv}")
