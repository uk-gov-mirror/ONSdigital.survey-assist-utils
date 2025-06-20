"""Module contains functionality to preprocess json output from Survey Assist."""

import json
import logging
import os
from datetime import datetime

import pandas as pd
import toml


class JsonPreprocessor:
    """Handles the processing of raw LLM JSON files into a clean DataFrame."""

    def __init__(self, config_path: str):
        """Initializes the preprocessor by loading the TOML configuration."""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str):
        """Loads configuration settings from a .toml file."""
        try:
            with open(config_path, encoding="utf-8") as file:
                configuration = toml.load(file)
                return configuration
        except FileNotFoundError:
            logging.error("Configuration file not found at '%s'", config_path)
            raise

    def get_local_filepaths(self) -> list[str]:
        """Gets a list of local filepaths to process based on config."""
        directory = self.config["paths"]["local_json_dir"]
        date_str = self.config["parameters"]["date_since"]
        given_date = datetime.strptime(date_str, "%Y%m%d")

        later_files = []
        logging.info("Searching for files in '%s' on or after %s", directory, date_str)
        try:
            for filename in os.listdir(directory):
                if filename.endswith("_output.json"):
                    try:
                        file_date = datetime.strptime(filename[:8], "%Y%m%d")
                        if file_date >= given_date:
                            later_files.append(os.path.join(directory, filename))
                    except ValueError:
                        continue
            logging.info("Found %d files to process.", len(later_files))
            return later_files
        except FileNotFoundError:
            logging.error("Source directory not found: %s", directory)
            return []

    def record_count(self, file_path: str) -> int:
        """Reads LLM response JSON data from a file, flattens it into a Pandas DataFrame.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            num_records (int): A count of the records expected.

        """
        num_records = 0
        try:
            with open(file_path, encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return num_records

        # Ensure json_data is a list of records
        if isinstance(json_data, dict):
            records_to_process = [
                json_data
            ]  # Handle case where file contains a single JSON object
        elif isinstance(json_data, list):
            records_to_process = json_data
        else:
            print("Error: JSON content is not a list or a single object (dictionary).")
            return num_records

        num_records = len(records_to_process)
        return num_records

    def flatten_llm_json_to_dataframe(
        self, file_path: str, max_candidates: int = 5
    ) -> pd.DataFrame:
        """Reads LLM response JSON data from a file, flattens it into a Pandas DataFrame.

        Args:
            file_path (str): The path to the JSON file.
            max_candidates (int): The maximum number of SIC candidates to flatten per record.

        Returns:
            pd.DataFrame: A Pandas DataFrame with the flattened JSON data.
        """
        all_flat_records = []
        print("file_path", file_path)

        try:
            with open(file_path, encoding="utf-8") as f:
                json_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {file_path}: {e}")
            return pd.DataFrame()

        # Ensure json_data is a list of records
        if isinstance(json_data, dict):
            records_to_process = [
                json_data
            ]  # Handle case where file contains a single JSON object
        elif isinstance(json_data, list):
            records_to_process = json_data
        else:
            print("Error: JSON content is not a list or a single object (dictionary).")
            return pd.DataFrame()

        print("records_to_process", len(records_to_process))
        for record in records_to_process:
            flat_record = {}

            # 1. Add top-level fields
            flat_record["unique_id"] = record.get("unique_id")
            flat_record["classified"] = record.get("classified")
            flat_record["followup"] = record.get("followup")
            # Rename top-level sic_code & sic_description to avoid clashes with candidate fields
            flat_record["chosen_sic_code"] = record.get("sic_code")
            flat_record["chosen_sic_description"] = record.get("sic_description")
            flat_record["reasoning"] = record.get("reasoning")

            # 2. Flatten request_payload
            payload = record.get(
                "request_payload", {}
            )  # Default to empty dict if payload is missing
            flat_record["payload_llm"] = payload.get("llm")
            flat_record["payload_type"] = payload.get("type")
            flat_record["payload_job_title"] = payload.get("job_title")
            flat_record["payload_job_description"] = payload.get("job_description")
            flat_record["payload_industry_descr"] = payload.get("industry_descr")

            # 3. Flatten sic_candidates
            candidates = record.get("sic_candidates", [])  # Default to empty list
            for i in range(max_candidates):
                if i < len(candidates) and isinstance(candidates[i], dict):
                    candidate = candidates[i]
                    flat_record[f"candidate_{i+1}_sic_code"] = candidate.get("sic_code")
                    flat_record[f"candidate_{i+1}_sic_descriptive"] = candidate.get(
                        "sic_descriptive"
                    )
                    flat_record[f"candidate_{i+1}_likelihood"] = candidate.get(
                        "likelihood"
                    )
                else:
                    # Fill with None if fewer than max_candidates or data is malformed
                    flat_record[f"candidate_{i+1}_sic_code"] = None
                    flat_record[f"candidate_{i+1}_sic_descriptive"] = None
                    flat_record[f"candidate_{i+1}_likelihood"] = None

            all_flat_records.append(flat_record)

        df = pd.DataFrame(all_flat_records)

        return df

    def count_all_records(self) -> int:
        """Main method to load, flatten, and combine all specified JSON files."""
        filepaths = self.get_local_filepaths()
        total_count = 0
        print("filepaths", filepaths)
        print(len(filepaths))
        if not filepaths:
            logging.warning("No files found to process.")
            return total_count

        for path in filepaths:
            count_length = self.record_count(path)
            total_count = total_count + count_length

        return total_count

    def process_files(self) -> pd.DataFrame:
        """Main method to load, flatten, and combine all specified JSON files."""
        filepaths = self.get_local_filepaths()
        print("filepaths", filepaths)
        print(len(filepaths))
        if not filepaths:
            logging.warning("No files found to process. Returning empty DataFrame.")
            return pd.DataFrame()

        all_dfs = [self.flatten_llm_json_to_dataframe(path) for path in filepaths]

        # Filter out empty dataframes that may result from processing errors
        all_dfs = [df for df in all_dfs if not df.empty]

        if not all_dfs:
            return pd.DataFrame()

        combined_df = pd.concat(all_dfs, ignore_index=True)
        # logging.info("Combined all files into a DataFrame with shape %s", combined_df.shape)

        unique_id_col = self.config["json_keys"]["unique_id"]
        combined_df.drop_duplicates(subset=[unique_id_col], inplace=True)
        # logging.info("Shape after dropping duplicates: %s", combined_df.shape)

        return combined_df

    def merge_eval_data(self, flattened_json_df):
        """Merges evaluation data from a CSV file with a flattened JSON DataFrame on
        the 'unique_id' column.

        The method reads a processed CSV file specified in the configuration, removes
        duplicate entries based on the 'unique_id' column, and performs an inner join
        with the provided DataFrame. The resulting merged DataFrame includes suffixes
        to distinguish overlapping column names.

        Parameters:
            flattened_json_df (pd.DataFrame): A DataFrame containing flattened JSON
            data with a 'unique_id' column.

        Returns:
            pd.DataFrame: A merged DataFrame containing rows with matching 'unique_id'
            values from both sources.
        """
        processed_csv_output = self.config["paths"]["processed_csv_output"]

        eval_data = pd.read_csv(processed_csv_output, dtype={"SIC_Division": str})
        eval_data = eval_data.drop_duplicates(subset="unique_id")
        merged_df = pd.merge(
            eval_data,
            flattened_json_df,
            on="unique_id",
            how="inner",
            suffixes=("_df1", "_df2"),
        )
        print(merged_df.shape)
        return merged_df
