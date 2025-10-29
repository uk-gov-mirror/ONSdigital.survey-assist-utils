#!/usr/bin/env python
"""Post-process OTPs to replace '-' with ' ',
rename columns and split into retain/distribute files.
"""
from argparse import ArgumentParser as AP

import pandas as pd
from urllib3.exceptions import HTTPError

from survey_assist_utils.logging import get_logger

LOG_LEVEL = "DEBUG"

parser = AP(description="Post-process OTPs to add missing columns and format data.")
parser.add_argument(
    "gcp_bucket_url", type=str, help="GCP Bucket URL where the OTP CSV file is stored."
)
parser.add_argument(
    "number_to_share", type=int, help="Number of OTPs to share with the PfR team."
)
parser.add_argument(
    "--overwrite",
    "-o",
    action="store_true",
    help="Overwrite existing output files if they already exist.",
)


def check_output_file_does_not_exist(bucket_url: str, logger) -> None:
    """Checks if the output files already exist in the GCP bucket."""
    retain_file_url = bucket_url.replace(".csv", "_ONS.csv")
    distribute_file_url = bucket_url.replace(".csv", "_PfR.csv")

    logger.info(f"Checking if output file already exists: {retain_file_url}")
    try:
        pd.read_csv(retain_file_url, nrows=0)
        logger.error(f"Retain file {retain_file_url} already exists.")
        raise FileExistsError(f"Retain file {retain_file_url} already exists.")
    except FileNotFoundError:
        logger.info(f"Retain file {retain_file_url} does not exist, proceeding.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking retain file: {e}")
        raise

    logger.info(f"Checking if output file already exists: {distribute_file_url}")
    try:
        pd.read_csv(distribute_file_url, nrows=0)
        logger.error(f"Retain file {distribute_file_url} already exists.")
        raise FileExistsError(f"Retain file {distribute_file_url} already exists.")
    except FileNotFoundError:
        logger.info(f"Retain file {distribute_file_url} does not exist, proceeding.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while checking retain file: {e}")
        raise


def get_otps_from_gcp(
    gcp_bucket_url: str, number_to_share: int, logger
) -> pd.DataFrame:
    """Reads OTPs from a CSV file located at the specified GCP Bucket URL."""
    logger.info(f"Reading OTPs from {gcp_bucket_url}")
    try:
        df = pd.read_csv(gcp_bucket_url)
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except HTTPError as e:
        logger.error(
            f"HTTP error occurred, suspect gcloud (re-)authentication required: {e}"
        )
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise
    if len(df) == 0:
        logger.error("No OTPs found in the provided GCP Bucket URL.")
        raise ValueError("No OTPs found.")
    if len(df) < number_to_share:
        logger.error(
            f"Requested number of OTPs to share ({number_to_share}) "
            f"exceeds available OTPs ({len(df)})."
        )
        raise ValueError("Insufficient OTPs available.")
    logger.info(f"Successfully read {len(df)} OTPs.")
    csv_columns = df.columns.tolist()
    logger.debug(f"Columns in OTPs DataFrame: {csv_columns}")
    if not {"survey_access_id", "one_time_passcode"}.issubset(set(csv_columns)):
        logger.error(
            f"The OTPs CSV file does not contain the required columns. "
            f"expected columns: 'survey_access_id', 'one_time_passcode'. "
            f"Found columns: {csv_columns}"
        )
        raise ValueError(
            "Missing required columns ('survey_access_id', 'one_time_passcode') in OTPs CSV."
        )
    return df


def reformat_data(df: pd.DataFrame, logger) -> pd.DataFrame:
    """Reformats the OTPs DataFrame by replacing '-' with ' ' and renaming columns."""
    logger.info("Reformatting OTPs data.")
    df["one_time_passcode"].replace("-", " ", inplace=True)
    df.rename(
        columns={
            "survey_access_id": "ONS Participant ID",
            "one_time_passcode": "PFR ID",
        },
        inplace=True,
    )
    logger.info("Reformatted OTPs data.")
    return df


def split_and_save_otps(
    df: pd.DataFrame, number_to_share: int, bucket_url: str, logger
) -> None:
    """Splits the OTPs DataFrame into retain and distribute files and saves them as CSV."""
    logger.info("Splitting OTPs into retain and distribute files.")
    distribute_df = df.iloc[1 : number_to_share + 1].reset_index(drop=True)
    # Note: we are keeping the first row (STP0000) in the retained collection.
    retain_df = pd.concat((df.iloc[:1], df.iloc[number_to_share + 1 :])).reset_index(
        drop=True
    )

    retain_file_url = bucket_url.replace(".csv", "_ONS.csv")
    distribute_file_url = bucket_url.replace(".csv", "_PfR.csv")

    logger.info(f"Saving retained OTPs to {retain_file_url}")
    try:
        retain_df.to_csv(retain_file_url, index=False)
        logger.info(f"Saved {len(retain_df)} OTPs to {retain_file_url}.")
    except Exception as e:
        logger.error(f"Failed to save retained OTPs: {e}")
        raise

    logger.info(f"Saving PfR OTPs to {distribute_file_url}")
    try:
        distribute_df.to_csv(distribute_file_url, index=False)
        logger.info(f"Saved {len(distribute_df)} OTPs to {distribute_file_url}.")
    except Exception as e:
        logger.error(f"Failed to save PfR OTPs: {e}")
        raise


if __name__ == "__main__":
    args = parser.parse_args()
    otp_logger = get_logger("post_process_OTPs", level=LOG_LEVEL.upper())
    if not args.overwrite:
        otp_logger.info("Checking for existing output files before proceeding.")
        check_output_file_does_not_exist(args.gcp_bucket_url, otp_logger)
    otp_logger.info("Starting OTP post-processing...")
    otp_logger.info("Starting to retrieve OTPs from GCP...")
    otps_df = get_otps_from_gcp(args.gcp_bucket_url, args.number_to_share, otp_logger)
    otp_logger.info("Reformatting OTPs data...")
    otps_df = reformat_data(otps_df, otp_logger)
    otp_logger.info("Splitting and saving OTPs...")
    split_and_save_otps(otps_df, args.number_to_share, args.gcp_bucket_url, otp_logger)
