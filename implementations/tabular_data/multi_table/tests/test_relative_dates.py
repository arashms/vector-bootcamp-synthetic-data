"""Relative-date encoding: real round trip, clipping, and cache separation."""

import json

import pandas as pd
import pytest

from implementations.tabular_data.multi_table.experiments import tuning
from implementations.tabular_data.multi_table.experiments.dataset_settings import DATA_DIR


def _real_frames():
    meta = json.loads((DATA_DIR / "dataset_meta.json").read_text())
    return {t: pd.read_csv(DATA_DIR / f"{t}.csv") for t in meta["tables"]}


@pytest.mark.skipif(not (DATA_DIR / "preprocess_meta.json").exists(), reason="processed cohort not prepared")
def test_real_dates_round_trip_and_never_precede_opening():
    """Real offsets are nonnegative and decoding restores the processed CSVs exactly."""
    frames = _real_frames()
    shifts = tuning.date_shifts(DATA_DIR)
    assert shifts == {"trans": 0, "loan": 212, "card": 457}
    encoded = tuning.encode_relative_dates(frames, shifts)
    for table, column in tuning.RELATIVE_DATE_COLUMNS.items():
        assert (encoded[table][column] >= 0).all()
    assert encoded["account"] is frames["account"]
    decoded = dict(encoded)
    assert tuning.decode_relative_dates(decoded, shifts) == {"trans": 0, "loan": 0, "card": 0}
    for table in tuning.RELATIVE_DATE_COLUMNS:
        pd.testing.assert_frame_equal(decoded[table], frames[table])


@pytest.mark.skipif(not (DATA_DIR / "preprocess_meta.json").exists(), reason="processed cohort not prepared")
def test_real_window_fractions_round_trip_within_unit_interval():
    """Window fractions of real data lie in [0, 1] and decode back to the CSV dates."""
    frames = _real_frames()
    shifts = tuning.date_shifts(DATA_DIR)
    end = tuning.data_end(DATA_DIR, shifts)
    assert end == 2188
    encoded = tuning.encode_relative_dates(frames, shifts, end)
    for table, column in tuning.RELATIVE_DATE_COLUMNS.items():
        assert encoded[table][column].between(0, 1).all()
    decoded = dict(encoded)
    assert tuning.decode_relative_dates(decoded, shifts, end) == {"trans": 0, "loan": 0, "card": 0}
    for table, column in tuning.RELATIVE_DATE_COLUMNS.items():
        pd.testing.assert_series_equal(decoded[table][column], frames[table][column].astype(float), atol=1e-9)


def test_window_decode_keeps_dates_between_opening_and_end():
    """Out-of-range fractions are clipped; late-opened accounts keep that date."""
    frames = {
        "account": pd.DataFrame({"account_id": [1, 2], "account_date": [100.0, 3000.0]}),
        "disp": pd.DataFrame({"disp_id": [7], "account_id": [1]}),
        "trans": pd.DataFrame({"account_id": [1, 1, 2], "trans_date": [-0.5, 1.5, 0.5]}),
        "loan": pd.DataFrame({"account_id": [1], "loan_date": [0.5]}),
        "card": pd.DataFrame({"disp_id": [7], "issued": [1.0]}),
    }
    shifts = {"trans": 0, "loan": 212, "card": 457}
    assert tuning.decode_relative_dates(frames, shifts, 1100) == {"trans": 2, "loan": 0, "card": 0}
    assert frames["trans"]["trans_date"].tolist() == [100.0, 1100.0, 3000.0]
    assert frames["loan"]["loan_date"].tolist() == [600.0 - 212]
    assert frames["card"]["issued"].tolist() == [1100.0 - 457]


def test_decode_clips_negative_offsets_to_opening_date():
    """Synthetic offsets below zero land on the account's opening date, never before."""
    frames = {
        "account": pd.DataFrame({"account_id": [1, 2], "account_date": [100.0, 300.0]}),
        "disp": pd.DataFrame({"disp_id": [7], "account_id": [2]}),
        "trans": pd.DataFrame({"account_id": [1, 2], "trans_date": [-5.0, 10.0]}),
        "loan": pd.DataFrame({"account_id": [2], "loan_date": [0.0]}),
        "card": pd.DataFrame({"disp_id": [7], "issued": [-1.0]}),
    }
    shifts = {"trans": 0, "loan": 212, "card": 457}
    assert tuning.decode_relative_dates(frames, shifts) == {"trans": 1, "loan": 0, "card": 1}
    assert frames["trans"]["trans_date"].tolist() == [100.0, 310.0]
    assert frames["loan"]["loan_date"].tolist() == [300.0 - 212]
    assert frames["card"]["issued"].tolist() == [300.0 - 457]


def test_switch_changes_every_stage_id_and_rejects_unknown_values():
    """Relative and absolute runs never share clustering, models, or samples."""
    config = tuning.base_config()
    assert config["date_representation"] == "fraction_of_window"
    assert tuning.base_config(fresh_training=False)["date_representation"] == "absolute"
    config = {**config, "date_representation": "absolute"}
    relative = {**config, "date_representation": "relative_to_account"}
    absolute_ids = tuning.stage_ids(config, "ctx")
    relative_ids = tuning.stage_ids(tuning.validate_config(relative), "ctx")
    assert all(absolute_ids[k] != relative_ids[k] for k in absolute_ids)
    with pytest.raises(ValueError, match="date_representation"):
        tuning.validate_config({**config, "date_representation": "days"})
