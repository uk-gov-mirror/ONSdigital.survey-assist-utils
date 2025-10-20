"""Record latency of a web request to BigQuery."""

import argparse
import datetime
import logging
import os
import random
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from export_to_bq import schema_entry, write_to_bq

from survey_assist_utils.api_token.jwt_utils import (  # pylint: disable=C0411
    generate_jwt,
)

env_path = Path(__file__).resolve().parent.parent.parent / ".env"
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


def get_env_var(
    name: str,
    required: bool = True,
    default: str | None = None,
    logger_tool: logging.Logger | None = None,
) -> str | None:
    """Get an environment variable, log an error if it's required and not set."""
    value = os.getenv(name, default)
    if required and (value is None) and logger_tool:
        logger_tool.error(f"Required environment variable {name} not set")
    return value


# Set up constants
PROJECT_ID = get_env_var(name="PROJECT_ID", required=True)
API_GATEWAY = get_env_var(name="API_GATEWAY", required=True)
SA_EMAIL = get_env_var(name="SA_EMAIL", required=True)
ENDPOINT = "/v1/survey-assist/classify"
BQ_DATASET_ID = get_env_var(name="BQ_DATASET_ID", required=True)
TABLE_ID = get_env_var(name="TABLE_ID", required=True)
TABLE_NAME = f"{PROJECT_ID}.{BQ_DATASET_ID}.{TABLE_ID}"
LOG_LEVEL = "DEBUG"

# Define schema
schema = [
    schema_entry("test_id", "INTEGER", "REQUIRED", "Unique test ID"),
    schema_entry("test_description", "STRING", "REQUIRED", "Test description."),
    schema_entry("test_timestamp", "DATETIME", "REQUIRED", "Test timestamp (UTC+0)."),
    schema_entry("step", "INTEGER", "REQUIRED", "Step number."),
    schema_entry(
        "total_latency", "FLOAT", "REQUIRED", "Total API call latency (in seconds)"
    ),
    schema_entry(
        "response_time", "FLOAT", "REQUIRED", "Request response time (in seconds)"
    ),
    schema_entry("status_code", "STRING", "REQUIRED", "HTTP status code."),
    schema_entry("job_title", "STRING", "REQUIRED", "Job title."),
    schema_entry("job_description", "STRING", "REQUIRED", "Job description."),
    schema_entry("org_description", "STRING", "REQUIRED", "Organisation description."),
    schema_entry(
        "classified", "INTEGER", "REQUIRED", "Classification success or failure."
    ),
    schema_entry("followup", "STRING", "REQUIRED", "Followup question."),
    schema_entry("code", "STRING", "REQUIRED", "Assigned SIC code."),
]


def prepare_auth(api_gateway_url: str, sa_email: str, logger_tool: logging.Logger):
    """Generates a JWT token and forms authorisation headers."""
    logger_tool.debug("Generating api token...")
    try:
        token = generate_jwt(
            audience=api_gateway_url, sa_email=sa_email, expiry_length=3600
        )
    except Exception as e:
        logger_tool.error(f"Error generating api token: {e}")
        raise
    auth_headers = {"Authorization": f"Bearer {token}"}
    return auth_headers


def prepare_payload(fake_data_csv: str, logger_tool: logging.Logger) -> dict:
    """Reads a random line from a CSV and prepares the request payload."""
    try:
        logger_tool.debug(f"Reading fake responses csv file: '{fake_data_csv}'")
        with open(fake_data_csv, encoding="utf-8") as file:
            lines = file.readlines()

        # The first line is the header
        line_count = len(lines)
        logger_tool.debug(f"Found {line_count} lines in '{fake_data_csv}'.")
        chosen_row_index = random.randint(1, line_count - 1)  # noqa: S311
        payload_data = lines[chosen_row_index].strip()
        logger_tool.debug(f"Selected line {chosen_row_index} for payload.")
        payload = {
            "llm": "gemini",
            "type": "sic",
            "job_title": "string",
            "job_description": "string",
            "org_description": "string",
            "options": {"sic": {"rephrased": "true"}, "soc": {"rephrased": "false"}},
        }

        try:
            j_title, j_desc, ind_desc = payload_data.split("|")
            payload["job_title"] = j_title
            payload["job_description"] = j_desc
            payload["org_description"] = ind_desc
            logger_tool.info("Successfully prepared payload from CSV.")
            return payload
        except ValueError as e:
            logger_tool.error(
                f"Wrong number of fields in CSV row: '{payload_data}'. Error: {e}"
            )
            raise
    except FileNotFoundError as e:
        logger_tool.error(f"Error: File '{fake_data_csv}' not found. {e}")
        raise
    except Exception as e:
        logger_tool.error(f"An unexpected error occurred in prepare_payload: {e}")
        raise


def step_1(  # pylint: disable=R0917,R0913 # noqa: PLR0913
    test_id: int,
    test_description: str,
    test_timestamp: str,
    logger_tool: logging.Logger,
    payload: dict,
    headers: dict,
):
    """Step 1: Measure latency of a web request."""
    logger_tool.debug(f"Step 1: Initiating API call to {API_GATEWAY}{ENDPOINT}")
    start = time.time()
    response = requests.post(
        f"{API_GATEWAY}{ENDPOINT}", json=payload, headers=headers, timeout=60
    )
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        logger_tool.error(
            f"Step 1: API call failed with status code {response.status_code}: {e}"
        )
        raise
    logger_tool.debug(
        f"Step 1: API call completed with status code {response.status_code}"
    )
    response_time = response.elapsed.total_seconds()
    end = time.time()
    latency = end - start
    result = {
        "test_id": [test_id],
        "test_description": [test_description],
        "test_timestamp": [test_timestamp],
        "step": [1],
        "total_latency": [latency],
        "response_time": [response_time],
        "status_code": [f"{response.status_code}"],
        "job_title": [payload["job_title"]],
        "job_description": [payload["job_description"]],
        "org_description": [payload["org_description"]],
        "classified": [int(response.json()["results"][0]["classified"])],
        "followup": [response.json()["results"][0]["followup"]],
        "code": [f"{response.json()["results"][0]["code"]}"],
    }
    logger_tool.debug("Step 1: Preparing to write results to BigQuery.")
    write_to_bq(
        result,
        logger_tool,
        schema,
        gcp_kwargs={
            "project_id": PROJECT_ID,
            "url": API_GATEWAY,
            "table_name": TABLE_NAME,
        },
    )
    logger_tool.debug("Step 1: Results written to BigQuery.")


def setup_logger():
    """Set up the logger."""
    logger_tool = logging.getLogger("loadrunner")
    logger_tool.handlers.clear()
    logger_tool.setLevel(LOG_LEVEL.upper())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger_tool.addHandler(sh)
    return logger_tool


def main(args, logger_tool):
    """Run a single latency test and store the result in BigQuery."""
    logger_tool.debug(f"Starting test {args.test_id}: {args.test_description}")
    # wait until the specified sync time
    # Parse sync_time as UTC+0
    sync_time = datetime.datetime.strptime(args.test_sync_timestamp, "%H:%M:%S").time()
    request_payload = prepare_payload("fake_data/fake_responses.csv", logger_tool)
    request_headers = prepare_auth(API_GATEWAY, SA_EMAIL, logger_tool)
    now = datetime.datetime.now(datetime.timezone.utc)
    time_to_wait = (
        datetime.datetime.combine(now.date(), sync_time, tzinfo=datetime.timezone.utc)
        - now
    ).total_seconds()
    logger_tool.debug(
        f"Waiting for sync time {sync_time} (in {time_to_wait:.2f} " "seconds)..."
    )
    time.sleep(time_to_wait)
    start_time = datetime.datetime.now(datetime.timezone.utc)
    logger_tool.debug("Starting loadrunner run...")
    logger_tool.debug(f"Test ID: {args.test_id}")
    logger_tool.debug(f"Test description: {args.test_description}")
    logger_tool.debug(f"Test start time (UTC+0): {start_time.isoformat()}")
    logger_tool.debug("Running step 1:...")
    step_1(
        args.test_id,
        args.test_description,
        start_time,
        logger_tool,
        request_payload,
        request_headers,
    )
    logger_tool.debug("Step 1 complete.")
    logger_tool.debug("loadrunner run complete.")


def setup_parser():
    """Set up the argument parser."""
    _parser = argparse.ArgumentParser(
        description=("loadrunner: a lightweight utility to execute system loads."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    _parser.add_argument(
        "test_id", type=int, help="The unique identifier for the test to be executed."
    )
    _parser.add_argument(
        "test_description",
        type=str,
        help="The brief description of the test to be executed.",
    )
    _parser.add_argument(
        "test_sync_timestamp",
        type=str,
        help="The synchronization timestamp for the test (UTC+0), in HH:MM:SS format.",
    )
    return _parser


if __name__ == "__main__":
    # set up the parser
    parser = setup_parser()
    # set up the logger
    logger = setup_logger()
    # parse the arguments
    arguments = parser.parse_args()
    # run the main function
    main(arguments, logger)
