"""Account cohort integrity, complete histories, and non-destructive preparation."""
import json
from copy import deepcopy

import pandas as pd
import pytest

from implementations.tabular_data.multi_table.data_preprocessing.account_sampling import (
    prepare_account_sample,
    sample_account_tables,
)


def test_selection_retains_complete_histories_and_every_required_parent(raw_tables):
    """Cohort links remain valid, including districts referenced only by clients."""
    original = deepcopy(raw_tables)
    cohort, info = sample_account_tables(raw_tables, 2, seed=42)
    ids = set(cohort["account"].account_id)
    expected = raw_tables["trans"].loc[raw_tables["trans"].account_id.isin(ids), "trans_id"]
    assert set(cohort["trans"].trans_id) == set(expected)
    assert len(cohort["trans"]) == 4
    assert info["overdraft_rows"] == 2
    assert set(cohort["client"].client_id) == set(cohort["disp"].client_id)
    assert set(cohort["card"].disp_id) <= set(cohort["disp"].disp_id)
    expected_districts = set(cohort["client"].district_id) | set(cohort["account"].district_id)
    assert set(cohort["district"].A1) == expected_districts
    again, again_info = sample_account_tables(raw_tables, 2, seed=42)
    assert info == again_info
    for table in raw_tables:
        pd.testing.assert_frame_equal(raw_tables[table], original[table])
        pd.testing.assert_frame_equal(cohort[table], again[table])


@pytest.mark.parametrize("size", [0, -1, 5, True])
def test_invalid_sample_size_fails(raw_tables, size):
    """Never silently change an invalid account sampling request."""
    with pytest.raises(ValueError, match="account_sample_size"):
        sample_account_tables(raw_tables, size)


def test_prepare_reuses_transforms_and_preserves_existing_files(tmp_path, raw_tables):
    """Preparation records provenance, preserves raw inputs, and rejects overwrites."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, frame in raw_tables.items():
        frame.to_csv(raw / f"{name}.csv", sep=";", index=False)
    before = {p.name: p.read_bytes() for p in raw.iterdir()}
    output = tmp_path / "cohort"
    result = prepare_account_sample(raw, output, 2, seed=42)
    assert result["n_rows"]["trans"] == 4
    assert not result["reused"]
    meta = json.loads((output / "preprocess_meta.json").read_text())
    assert meta["trans_sample_size"] is None
    assert meta["account_sampling"]["all_transactions_retained"]
    assert len(json.loads((output / "dataset_meta.json").read_text())["tables"]) == 8
    assert prepare_account_sample(raw, output, 2, seed=42)["reused"]
    with pytest.raises(ValueError, match="different inputs/settings"):
        prepare_account_sample(raw, output, 3, seed=42)
    (output / "trans.csv").write_text((output / "trans.csv").read_text() + "\n")
    with pytest.raises(ValueError, match="changed or are missing"):
        prepare_account_sample(raw, output, 2, seed=42)
    assert before == {p.name: p.read_bytes() for p in raw.iterdir()}
    with pytest.raises(ValueError, match="separate processed"):
        prepare_account_sample(raw, raw, 2)
    with pytest.raises(ValueError, match="already exists"):
        prepare_account_sample(raw, tmp_path, 2)
