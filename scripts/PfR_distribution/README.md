# post_process_OTPs

Post-process OTPs to add missing columns, normalise passcodes and split the file into retained and distributed outputs for PfR.

## Description

This utility reads a CSV of one-time passcodes (OTPs) from a GCP Bucket, validates the data, replaces hyphens in passcodes with spaces, renames columns for downstream consumers, and writes two CSV outputs to the same folder in the GCP Bucket:
- a retained ONS file (suffix `_ONS.csv`)
- a PfR distribution file (suffix `_PfR.csv`)

The script keeps the first row (STP0000) in the retained set and selects the requested number of OTP rows for distribution.

## Usage

usage: `python post_process_OTPs.py [-h] [--overwrite] <gcp_bucket_url> <number_to_share>`

Positional arguments:

  `gcp_bucket_url`  - GCP Bucket URL where the OTP CSV file is stored.

  `number_to_share` - Number of OTPs to share with the PfR team.

Options:

  `-h`, `--help`      - show this help message and exit

  `--overwrite`, `-o` - Overwrite existing output files if they already exist.

Example:
```
python post_process_OTPs.py gs://my-bucket/path/otps.csv 10
```
To force overwrite of existing outputs:
```
python post_process_OTPs.py --overwrite gs://my-bucket/path/otps.csv 10
```

## Expected input CSV

The input CSV must include the following columns:
- `survey_access_id`
- `one_time_passcode`

The script will:
- replace `-` characters in `one_time_passcode` with spaces,
- rename columns to `ONS Participant ID` and `PFR ID`.

## Outputs

Given input `example.csv` the script writes:
- `.example_ONS.csv` — retained OTPs (first / `STP0000` row kept plus remaining non-shared rows)
- `example_PfR.csv` — OTPs selected for sharing (the requested number, sequentially starting with `STP0001`)

Both files are written to the same location as the input path (so the path must be writable).

## Requirements

- Python 3.8+
- pandas
- Access to the GCP path: appropriate credentials/configuration so pandas can read the file (e.g. gcloud auth).

## Logging & errors

The script logs progress and errors. Common failure modes:
- Missing required columns in input CSV → script exits with a clear error.
- Requested share count larger than available OTPs → script exits with a clear error.
- Output files already exist → script aborts unless `--overwrite` is passed.

## Notes

- The script assumes the first row should be retained.
- Ensure the process running the script has read/write access to the GCP location or local path used.
