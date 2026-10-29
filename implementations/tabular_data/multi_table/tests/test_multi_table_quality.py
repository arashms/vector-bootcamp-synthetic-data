"""Full-schema diagnostics must detect damage outside the transaction table."""
import json
import pickle
from copy import deepcopy

import numpy as np
import pytest

from implementations.tabular_data.multi_table.data_preprocessing import pre_process_berka_all_tabels as preprocessing
from implementations.tabular_data.multi_table.evaluation.multi_table_quality import multi_table_quality
from implementations.tabular_data.multi_table.evaluation.transaction_evaluation import evaluate_saved_runs
from implementations.tabular_data.multi_table.experiments.tuning import data_signature


def score(real, generated):
    """Score the full fixture schema with production domain definitions."""
    domains = {name: preprocessing.domain_for_table(name) for name in real}
    return multi_table_quality(real, generated, preprocessing.DEFAULT_DATASET_META, domains)


def test_identity_covers_eight_tables_and_eight_edges(processed_tables):
    """Identical data has zero marginal/cardinality errors with all edges present."""
    summary, details = score(processed_tables, processed_tables)
    assert summary["all_table_count"] == 8
    assert summary["all_edge_count"] == 8
    assert summary["all_table_numerical_ks"] == 0
    assert summary["all_table_categorical_tvd"] == 0
    assert summary["all_cardinality_ks"] == 0
    assert summary["all_min_fk_valid_rate"] == 1
    assert summary["all_min_pk_unique_rate"] == 1
    assert len(details["multi_parent_scores"]) == 1
    assert details["multi_parent_scores"].iloc[0].relationship == "client->disp<-account"
    assert details["all_cross_table_correlations"].absolute_error.dropna().eq(0).all()


def test_non_transaction_damage_and_external_ids(processed_tables):
    """Loan distributions count, but arbitrary external identifier magnitudes do not."""
    changed = deepcopy(processed_tables)
    changed["trans"]["account"] += 10**9
    changed["order"]["account_to"] += 10**9
    summary, details = score(processed_tables, changed)
    assert summary["all_table_numerical_ks"] == 0
    assert set(details["excluded_features"].column) == {"account", "account_to"}
    changed["loan"]["amount"] *= 10
    summary, details = score(processed_tables, changed)
    assert summary["all_table_numerical_ks"] > 0
    assert "account" not in set(details["all_column_distributions"].query("table == 'trans'").column)


def test_orphans_duplicates_and_undefined_pairs_are_visible(processed_tables):
    """Bad keys remain visible in integrity scores and relationship coverage."""
    changed = deepcopy(processed_tables)
    changed["loan"].loc[0, "account_id"] = 999
    changed["client"].loc[1, "client_id"] = changed["client"].loc[0, "client_id"]
    summary, details = score(processed_tables, changed)
    assert summary["all_min_fk_valid_rate"] < 1
    assert summary["all_min_pk_unique_rate"] < 1
    loan_edge = details["all_edge_scores"].set_index("edge").loc["account->loan"]
    assert loan_edge.invalid_foreign_keys == 1
    assert loan_edge.synthetic_joined_rows == 3
    assert details["multi_parent_scores"].iloc[0].join_status == "invalid_parent_primary_key"
    assert summary["multi_parent_undefined_pairs"] > 0
    changed["client"]["client_id"] = changed["client"].client_id.astype(float)
    changed["client"].loc[0, "client_id"] = np.inf
    summary, _ = score(processed_tables, changed)
    assert summary["all_min_pk_unique_rate"] < 1


def test_schema_missing_table_is_rejected(processed_tables):
    """A partial dataset must not be labeled as full-schema quality."""
    changed = {name: frame for name, frame in processed_tables.items() if name != "card"}
    with pytest.raises(ValueError, match="synthetic tables missing"):
        score(processed_tables, changed)


def test_multi_parent_joint_categories_detect_rewiring(processed_tables):
    """Valid FK values alone do not ensure realistic parent combinations."""
    changed = deepcopy(processed_tables)
    changed["disp"]["account_id"] = [2, 1, 4, 3]
    summary, details = score(processed_tables, changed)
    assert summary["all_min_fk_valid_rate"] == 1
    assert summary["multi_parent_joint_categorical_tvd"] > 0
    assert details["multi_parent_scores"].iloc[0].synthetic_joined_rows == len(changed["disp"])


def test_shared_report_exports_both_views_and_rejects_other_data(tmp_path, processed_tables, raw_tables):
    """All-table checks are actually wired into the notebook-facing evaluation API."""
    data = tmp_path / "data"
    data.mkdir()
    epochs = {}
    for name, frame in processed_tables.items():
        frame.to_csv(data / f"{name}.csv", index=False)
        (data / f"{name}_domain.json").write_text(json.dumps(preprocessing.domain_for_table(name)))
        epochs[name] = preprocessing.preprocess_table(name, raw_tables[name], None)[2]
    (data / "dataset_meta.json").write_text(json.dumps(preprocessing.DEFAULT_DATASET_META))
    (data / "preprocess_meta.json").write_text(json.dumps({"date_epochs_yymmdd": epochs}))
    trial = tmp_path / "trial"
    trial.mkdir()
    artifact = trial / "synthetic_final_with_keys.pkl"
    artifact.write_bytes(pickle.dumps(processed_tables))
    original = artifact.read_bytes()
    (trial / "manifest.json").write_text(json.dumps({"context": {"data": data_signature(data)}}))
    comparison, report = evaluate_saved_runs(data, [trial], tmp_path / "reports", bootstrap_repetitions=0,
                                            random_forest_params={"n_estimators": 5, "n_jobs": 1})
    assert comparison.loc["trial", "all_table_count"] == 8
    assert comparison.loc["trial", "all_edge_count"] == 8
    assert comparison.loc["trial", "all_table_numerical_ks"] == 0
    assert "average_precision" in comparison
    assert (report / "trial/all_table_scores.csv").is_file()
    assert (report / "trial/multi_parent_scores.csv").is_file()
    assert (report / "multi_table_quality.png").is_file()
    assert artifact.read_bytes() == original
    loan = data / "loan.csv"
    loan.write_text(loan.read_text() + "\n")
    with pytest.raises(ValueError, match="different reference dataset"):
        evaluate_saved_runs(data, [trial], tmp_path / "reports")
