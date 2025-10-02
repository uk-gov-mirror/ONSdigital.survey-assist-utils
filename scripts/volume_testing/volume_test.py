"""Record latency of a web request to BigQuery."""
import argparse
import datetime
import sys
import time
import os
import logging
import requests
import pandas as pd
import pandas_gbq as pgbq

from .export_to_BQ import write_to_bq, schema_entry

# Set up constants
PROJECT_ID = os.getenv("PROJECT_ID")
API_GATEWAY_URL = os.getenv("API_GATEWAY_URL")
ENDPOINT = "/classify"
BQ_DATASET_ID = os.getenv("BQ_DATASET_ID")
TABLE_ID = os.getenv("TABLE_ID")
TABLE_NAME = f"{PROJECT_ID}.{BQ_DATASET_ID}.{TABLE_ID}"
LOG_LEVEL = "DEBUG"

# Define schema
schema = [
    schema_entry("test_id", "INTEGER", "REQUIRED", "Unique test ID"),
    schema_entry("test_description", "STRING", "REQUIRED", "Test description."),
    schema_entry("test_timestamp", "DATETIME", "REQUIRED", "Test timestamp (UTC+0)."),
    schema_entry("step", "INTEGER", "REQUIRED", "Step number."),
    schema_entry("latency", "FLOAT", "REQUIRED", "Total API call latency (in seconds)"),
    schema_entry("response_time", "FLOAT", "REQUIRED", "Request response time (in seconds)"),
]

headers = {
    "jwt": "TODO"
}

payload = {
    "job_title": "test",
    "job_description": "test",
    "industry_description": "test",
}

def step_1(test_id, test_description, test_timestamp, logger, payload, headers):
    """Step 1: Measure latency of a web request."""
    start = time.time()
    response = requests.post(f"{API_GATEWAY_URL}{ENDPOINT}", json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    response_time = response.elapsed.total_seconds()
    end = time.time()
    latency = end - start
    result = {
        "test_id": [test_id],
        "test_description": [test_description],
        "test_timestamp": [test_timestamp],
        "step": [1],
        "total_latency": [latency],
        "response_time": [response_time]
    }
    write_to_bq(result,
                logger,
                schema,
                gcp_kwargs={
                            'project_id': PROJECT_ID,
                            'url': API_GATEWAY_URL,
                            'table_name': TABLE_NAME
                           },
               )

def setup_logger():
    logger = logging.getLogger("loadrunner")
    logger.handlers.clear()
    logger.setLevel(LOG_LEVEL.upper())
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

def main(args, logger):
    """Run a single latency test and store the result in BigQuery."""
    logger.debug(f"Starting test {args.test_id}: {args.test_description}")
    # wait until the specified sync time
    # Parse sync_time as UTC+0
    sync_time = datetime.datetime.strptime(
        args.test_sync_timestamp, "%H:%M:%S"
    ).time()
    now = datetime.datetime.now(datetime.timezone.utc)
    time_to_wait = (
        datetime.datetime.combine(
            now.date(), sync_time, tzinfo=datetime.timezone.utc
        ) - now
    ).total_seconds()
    logger.debug(
        f"Waiting for sync time {sync_time} (in {time_to_wait:.2f} "
        "seconds)..."
    )
    time.sleep(time_to_wait)
    start_time = datetime.datetime.now(datetime.timezone.utc)
    logger.debug("Starting loadrunner run...")
    logger.debug(f"Test ID: {args.test_id}")
    logger.debug(f"Test description: {args.test_description}")
    logger.debug(f"Test start time (UTC+0): {start_time.isoformat()}")
    logger.debug("Running step 1:...")
    step_1(args.test_id, args.test_description, start_time, logger, payload, headers)
    logger.debug("Step 1 complete.")
    logger.debug("loadrunner run complete.")

def setup_parser():
    """Set up the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "loadrunner: a lightweight utility to execute system loads."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("test_id", type=int,
                        help="The unique identifier for the test to be executed.")
    parser.add_argument("test_description", type=str,
                        help="The brief description of the test to be executed.")
    parser.add_argument("test_sync_timestamp", type=str,
                        help="The synchronization timestamp for the test (UTC+0), in HH:MM:SS format.")
    return parser    

if __name__ == "__main__":
    # set up the parser
    parser = setup_parser()
    # set up the logger
    logger = setup_logger()
    # parse the arguments
    args = parser.parse_args()
    # run the main function
    main(args, logger)