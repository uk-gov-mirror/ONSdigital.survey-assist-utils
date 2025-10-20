"""BigQuery interaction utilities to support volume testing."""

import logging
import time
from typing import Optional

import pandas as pd
import pandas_gbq as pgbq


def schema_entry(name: str, datatype: str, mode: str, description: str) -> dict:
    """Create a BigQuery schema entry."""
    return {
        "name": name,
        "type": datatype,
        "mode": mode,
        "description": description,
    }


def _backoff(
    initial_wait: float = 3.0,
    backoff_factor: float = 1.5,
    max_attempts: int = 5,
    logger: Optional[logging.Logger] = None,
):
    """Generator for exponential backoff timing."""
    wait = initial_wait
    count = 1
    while True:
        if count > max_attempts:
            if logger:
                logger.error(  # type: ignore[union-attr]
                    f"Exceeded maximum attempts ({max_attempts}) for BigQuery write."
                )
            raise StopIteration(
                f"failed to write after maximum ({max_attempts}) attempts"
            )
        yield count, wait
        wait *= backoff_factor
        count += 1


def write_to_bq(
    result: dict,
    logger: logging.Logger,
    schema: list[dict],
    gcp_kwargs: dict,
) -> None:
    """Write the result to BigQuery."""
    df = pd.DataFrame(result)

    for attempt, wait_time in _backoff(logger=logger):
        try:
            pgbq.to_gbq(
                df,
                gcp_kwargs["table_name"],
                project_id=gcp_kwargs["project_id"],
                if_exists="append",
                table_schema=schema,
            )
            break  # exit retry loop on success
        except pgbq.exceptions.GenericGBQException as e:
            if (
                "Reason: 429 Exceeded rate limits: too many table update "
                "operations for this table" in str(e)
            ):
                logger.debug(
                    f"BQ write rate limit exceeded, waiting {wait_time:.3f}s  "
                    f"before retrying (attempt {attempt})..."
                )
                time.sleep(wait_time)
            else:
                raise e
    else:
        raise RuntimeError(
            "Failed to write to BigQuery after multiple attempts, all failed"
            "with rate limit errors."
        )
