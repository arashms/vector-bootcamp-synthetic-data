"""Training-only account feature: real counts, toolkit loading, and cache separation."""

import json

import pandas as pd
import pytest
from midst_toolkit.models.clavaddpm.data_loaders import load_tables

from implementations.tabular_data.multi_table.experiments import tuning
from implementations.tabular_data.multi_table.experiments.dataset_settings import DATA_DIR


@pytest.mark.skipif(not (DATA_DIR / "dataset_meta.json").exists(), reason="processed cohort not prepared")
def test_ever_overdrawn_loads_through_toolkit(tmp_path):
    """The flag matches the CPU check (28 of 500) and loads as a discrete column."""
    meta = json.loads((DATA_DIR / "dataset_meta.json").read_text())
    frames = {t: pd.read_csv(DATA_DIR / f"{t}.csv") for t in meta["tables"]}
    augmented = tuning.add_account_feature(frames, "ever_overdrawn")
    assert int(augmented["account"]["ever_overdrawn"].sum()) == 28
    assert "ever_overdrawn" not in frames["account"]
    source = tuning.feature_data_dir(DATA_DIR, tmp_path, "ever_overdrawn")
    tables, _, _ = load_tables(source, train_data=augmented)
    assert tables["account"].domain["ever_overdrawn"]["type"] == "discrete"
    assert "ever_overdrawn" in tables["account"].data
    assert "ever_overdrawn" not in json.loads((DATA_DIR / "account_domain.json").read_text())


def test_feature_switch_validates_and_separates_caches():
    """Unknown features fail; the flag changes stage IDs but 'none' keeps old IDs."""
    base = tuning.base_config()
    assert base["account_feature"] == "none"
    with pytest.raises(ValueError, match="account_feature"):
        tuning.validate_config({**base, "account_feature": "bogus"})
    without = {k: v for k, v in base.items() if k != "account_feature"}
    flagged = {**base, "account_feature": "ever_overdrawn"}
    assert tuning.stage_ids(without, "ctx") == tuning.stage_ids(base, "ctx")
    assert tuning.stage_ids(flagged, "ctx")["cluster"] != tuning.stage_ids(base, "ctx")["cluster"]
