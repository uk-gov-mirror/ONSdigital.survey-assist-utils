#!/usr/bin/env python
"""This script retrieves survey responses from a Google Firestore database,
processes the data by flattening nested structures, and saves the results
into a CSV file.

It processes the data in chunks to handle large datasets efficiently.
"""
import collections.abc
from argparse import ArgumentParser as AP
from collections.abc import Generator

import pandas as pd
from firebase_admin import firestore, initialize_app


def setup_parser() -> AP:
    """Sets up a CLI parser."""
    parser = AP("Utility to retrieve survey responses from a Firestore database.")
    parser.add_argument("project_id", type=str, help="The Google Cloud project ID.")
    parser.add_argument("database_id", type=str, help="The Firestore database ID.")
    parser.add_argument("collection_name", type=str, help="The collection_name.")
    parser.add_argument(
        "output_name", type=str, help="The name of the output CSV file."
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=5,
        help="The connection timeout in seconds.",
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=1000,
        help="The number of documents to process in each chunk.",
    )
    return parser


def flatten_dict(
    d: dict | collections.abc.MutableMapping, parent_key: str = "", sep: str = "_"
) -> dict:
    """Flattens a nested dictionary (or dict-like object).

    Nested dictionary keys are combined with a separator. Lists are expanded
    such that each element gets its own key with an index.

    Args:
        d (dict): The dictionary to flatten.
        parent_key (str): The base key for constructing new keys. Defaults to ''.
        sep (str): The separator to use between nested keys. Defaults to '_'.

    Returns:
        dict: The flattened dictionary.
    """
    items: list[tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Handle item being a dictionary:
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten_dict(v, parent_key=new_key, sep=sep).items())
        # Handle item being a list:
        elif isinstance(v, (list, tuple)):
            if (
                not v
            ):  # Handle empty lists by acting as if there's a single (empty) element:
                items.append((f"{new_key}{sep}0", None))
            else:
                for i, item in enumerate(v):
                    # Create a key for each list item - e.g. 'responses_0'
                    list_key = f"{new_key}{sep}{i}"
                    if isinstance(item, collections.abc.MutableMapping):
                        items.extend(flatten_dict(item, list_key, sep=sep).items())
                    else:
                        # If the item in the list is not a dictionary, just save it with its index.
                        items.append((list_key, item))
        # No need to handle as a special case if item is already a single value:
        else:
            items.append((new_key, v))
    return dict(items)


def connect_to_firestore(
    project_id: str,
    database_id: str,
    collection_name: str = "survey_results",
    timeout: int | float = 5,
):
    """Initializes a connection to a Firestore database and returns a collection.

    Args:
        project_id (str): The Google Cloud project ID.
        database_id (str): The Firestore database ID.
        collection_name (str): The name of the Firestore collection.
            Defaults to 'survey_results'.
        timeout (int | float): The HTTP timeout in seconds for the connection.
            Defaults to 5.

    Returns:
        google.cloud.firestore_v1.collection.CollectionReference: A reference to
            the specified collection.

    Raises:
        ValueError: If `timeout` is not a positive number.
        ConnectionError: If the connection to Firestore fails or the
            specified collection is not found.
    """
    if timeout <= 0:
        raise ValueError("`timeout` must be positive and non-zero.")

    # Initialize Firestore connection
    app_options = {
        "projectId": project_id,
        "httpTimeout": timeout,
    }
    app = initialize_app(options=app_options)
    db = firestore.client(app=app, database_id=database_id)

    # non-intrusive test to verify connection
    try:
        if collection_name not in [collection.id for collection in db.collections()]:
            raise ValueError(
                f"'{collection_name}' collection not found, or not accessible."
            )
    except Exception as e:
        raise ConnectionError(f"Error when connecting to Firestore: {e}") from e
    return db.collection(collection_name)


def chunker(db_collection, chunk_size: int = 1000) -> Generator[list[dict], None, None]:
    """Yields batches of documents from a Firestore collection.

    This generator function fetches documents from a Firestore collection in batches,
    ordered by 'time_end' to process large collections without loading everything into
    memory.

    Args:
        db_collection: The Firestore collection reference to query.
        chunk_size (int): The number of documents to retrieve in each chunk.
            Defaults to 1000.

    Yields:
        list[dict]: A list of flattened dictionaries, where each dictionary
            represents a survey response document.
    """
    if chunk_size <= 0:
        raise ValueError("`chunk_size` must be positive and non-zero.")
    query = db_collection
    last_doc = None
    while True:
        try:
            current_query = (
                query.order_by("time_end").order_by("__name__").limit(chunk_size)
            )
            if current_query.count().get()[0][0].value == 0:
                raise AttributeError(
                    "'time_end' is not an attribute of the query, ordering by ID instead."
                )
        except AttributeError:
            current_query = query.order_by("__name__").limit(chunk_size)
        if last_doc:
            current_query = current_query.start_after(last_doc)

        docs = list(current_query.stream())
        if len(docs) == 0:
            break

        last_doc = docs[-1]
        yield [flatten_dict({"id": doc.id} | doc.to_dict()) for doc in docs]


def process_and_save_survey_results(
    project_id: str,
    database_id: str,
    collection_name: str = "survey_results",
    output_name: str = "survey_results.csv",
    timeout: int | float = 30,
) -> None:
    """Connects to Firestore, processes survey results, and saves them to a CSV.

    This function orchestrates the full process of fetching survey data in
    chunks, flattening each document, and appending the results to a CSV file.

    Args:
        project_id (str): The Google Cloud project ID.
        database_id (str): The Firestore database ID.
        collection_name (str): The name of the Firestore collection to process.
        output_name (str): The name of the output CSV file.
        timeout (int): The connection timeout in seconds.
    """
    survey_results_collection = connect_to_firestore(
        project_id, database_id, collection_name, timeout
    )
    for chunk_id, results_chunk in enumerate(chunker(survey_results_collection)):
        df = pd.DataFrame(results_chunk)
        if chunk_id == 0:
            df.to_csv(output_name, index=False)
        else:
            df.to_csv(output_name, index=False, mode="a", header=False)


if __name__ == "__main__":
    cli_parser = setup_parser()
    args = cli_parser.parse_args()
    process_and_save_survey_results(
        args.project_id,
        args.database_id,
        args.collection_name,
        args.output_name,
        args.timeout,
    )
