"""Editable dataset defaults shared by preparation and all experiment notebooks.

Change the account count/seed to prepare another Berka cohort, or set DATA_DIR to
another already-preprocessed Berka-compatible dataset. Restart notebook kernels
after editing this file. Model/report reuse is isolated by actual data contents.
"""
from pathlib import Path


MULTI_TABLE = Path(__file__).resolve().parents[1]
ACCOUNT_SAMPLE_SIZE = 500  # Use None for all accounts.
DATA_SEED = 42
RAW_DATA_DIR = MULTI_TABLE / "data/berka/raw_data"
COHORT_LABEL = "all" if ACCOUNT_SAMPLE_SIZE is None else str(ACCOUNT_SAMPLE_SIZE)
DATA_DIR = MULTI_TABLE / "data" / f"berka_accounts_{COHORT_LABEL}_seed{DATA_SEED}"
# DATA_DIR may instead point to an existing processed dataset with the same schema.
