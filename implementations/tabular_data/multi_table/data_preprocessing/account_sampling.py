"""Select a complete account cohort and reuse the existing Berka preprocessing.

Preparation writes only to a new dataset directory. Complete identical datasets
can be reused after checking their provenance and artifact hashes.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

from implementations.tabular_data.multi_table.data_preprocessing import pre_process_berka_all_tabels as preprocessing
from implementations.tabular_data.single_table.data_processing import preprocess_berka_trans as common_preprocessing


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sample_account_tables(tables, account_sample_size=500, *, seed=42):
    """Choose accounts uniformly, then retain every linked raw row in the cohort.

    Clients are those referenced by selected dispositions; districts are the
    union referenced by selected accounts and clients. Dispositions to unselected
    accounts are excluded even if their client also has a selected account.
    """
    missing = set(preprocessing.TABLE_NAMES) - set(tables)
    if missing:
        raise ValueError(f"Missing raw tables: {sorted(missing)}")
    accounts = tables["account"]
    if accounts.account_id.isna().any() or accounts.account_id.duplicated().any():
        raise ValueError("Raw account IDs must be present and unique.")
    if account_sample_size is None:
        account_sample_size = len(accounts)
    if isinstance(account_sample_size, bool) or not isinstance(account_sample_size, int) or not 0 < account_sample_size <= len(accounts):
        raise ValueError(f"account_sample_size must be between 1 and {len(accounts)}, or None for all.")
    selected = accounts.sample(n=account_sample_size, random_state=seed)
    ids = set(selected.account_id)
    cohort = {"account": accounts[accounts.account_id.isin(ids)].copy()}
    for name in ("trans", "loan", "order", "disp"):
        cohort[name] = tables[name][tables[name].account_id.isin(ids)].copy()
    cohort["client"] = tables["client"][tables["client"].client_id.isin(cohort["disp"].client_id)].copy()
    cohort["card"] = tables["card"][tables["card"].disp_id.isin(cohort["disp"].disp_id)].copy()
    district_ids = set(cohort["account"].district_id) | set(cohort["client"].district_id)
    cohort["district"] = tables["district"][tables["district"].A1.isin(district_ids)].copy()
    _validate_cohort(cohort)
    negative = pd.to_numeric(cohort["trans"].balance, errors="raise").lt(0)
    info = {
        "sampling_unit": "account", "account_sample_size": account_sample_size, "seed": seed,
        "all_transactions_retained": True, "trans_sample_size": None,
        "source_n_rows": {name: len(tables[name]) for name in preprocessing.TABLE_NAMES},
        "n_rows": {name: len(cohort[name]) for name in preprocessing.TABLE_NAMES},
        "overdraft_rows": int(negative.sum()),
        "overdraft_accounts": int(cohort["trans"].loc[negative, "account_id"].nunique()),
        "overdraft_prevalence": float(negative.mean()),
        "selected_account_ids": sorted(int(value) for value in ids),
    }
    return cohort, info


def _validate_cohort(cohort):
    for name, frame in cohort.items():
        key = "A1" if name == "district" else f"{name}_id"
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"Raw {name} primary keys must be present and unique.")
    for parent, child in preprocessing.DEFAULT_DATASET_META["relation_order"]:
        if parent is None:
            continue
        foreign_key = f"{parent}_id"
        parent_key = "A1" if parent == "district" else foreign_key
        if not cohort[child][foreign_key].isin(cohort[parent][parent_key]).all():
            raise ValueError(f"Account selection left orphan rows on {parent} -> {child}.")
    empty = [name for name, frame in cohort.items() if frame.empty]
    if empty:
        raise ValueError(f"The selected cohort has empty tables: {empty}. Increase account_sample_size or change seed.")


def load_account_sample(raw_data_dir, account_sample_size=500, *, seed=42):
    """Preview a cohort in memory; create no dataset files."""
    raw_data_dir = Path(raw_data_dir)
    tables = {name: preprocessing.load_raw_table(preprocessing.find_table_path(raw_data_dir, name), name)
              for name in preprocessing.TABLE_NAMES}
    return sample_account_tables(tables, account_sample_size, seed=seed)


def prepare_account_sample(raw_data_dir, output_data_dir, account_sample_size=500, *, seed=42):
    """Write a separately versioned dataset or reuse an identical completed one.

    Existing untracked/partial/mismatched output directories are rejected, never
    overwritten. All raw transaction rows for selected accounts are retained.
    """
    raw_data_dir, output_data_dir = Path(raw_data_dir).resolve(), Path(output_data_dir).resolve()
    if raw_data_dir == output_data_dir or output_data_dir.is_relative_to(raw_data_dir):
        raise ValueError("Choose a separate processed output directory outside the raw data directory.")
    inputs = {
        "account_sample_size": account_sample_size, "seed": seed,
        "raw_files": {name: _hash_file(preprocessing.find_table_path(raw_data_dir, name))
                      for name in preprocessing.TABLE_NAMES},
        "preprocessing_source": {"account_sampling": _hash_file(__file__),
                                 "transforms": _hash_file(preprocessing.__file__),
                                 "common_transforms": _hash_file(common_preprocessing.__file__)},
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn")},
    }
    manifest_path = output_data_dir / "account_sample_manifest.json"
    if output_data_dir.exists():
        if not manifest_path.is_file():
            raise ValueError("Output directory already exists without a cohort manifest. Choose a new OUTPUT_DATA_DIR.")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("inputs") != inputs:
            raise ValueError("Output dataset is incomplete or has different inputs/settings. Choose a new OUTPUT_DATA_DIR.")
        artifacts = manifest.get("artifacts", {})
        required = {"preprocess_meta.json", "dataset_meta.json", "selected_accounts.csv"}
        required |= {f"{name}{suffix}" for name in preprocessing.TABLE_NAMES for suffix in (".csv", "_domain.json")}
        if not required.issubset(artifacts) or any(
            not (output_data_dir / name).is_file() or _hash_file(output_data_dir / name) != checksum
            for name, checksum in artifacts.items()
        ):
            raise ValueError("Saved cohort artifacts changed or are missing. Choose a new OUTPUT_DATA_DIR.")
        return {"output_dir": output_data_dir, "reused": True, **manifest["sample"]}
    cohort, info = load_account_sample(raw_data_dir, account_sample_size, seed=seed)
    output_data_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "inputs": inputs, "raw_data_dir": str(raw_data_dir), "sample": info}
    preprocessing.save_json(manifest_path, manifest)
    try:
        preprocessing.preprocess_berka_all_tables(raw_data_dir, output_data_dir, seed=seed, raw_tables=cohort)
        meta_path = output_data_dir / "preprocess_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["account_sampling"] = info
        preprocessing.save_json(meta_path, meta)
        pd.DataFrame({"account_id": info["selected_account_ids"]}).to_csv(output_data_dir / "selected_accounts.csv", index=False)
        manifest["artifacts"] = {path.name: _hash_file(path) for path in sorted(output_data_dir.iterdir())
                                 if path.is_file() and path != manifest_path}
        manifest["status"] = "complete"
        preprocessing.save_json(manifest_path, manifest)
    except BaseException as error:
        manifest.update(status="failed", error=repr(error))
        preprocessing.save_json(manifest_path, manifest)
        raise
    return {"output_dir": output_data_dir, "reused": False, **info}
