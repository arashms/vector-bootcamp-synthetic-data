"""Regression checks for transaction evaluation and controlled experiment reuse."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from implementations.tabular_data.multi_table.evaluation.overdraft_transfer_evaluator import (
    FEATURES,
    OverdraftTransferEvaluator,
    align_categories,
    classification_metrics,
)
from implementations.tabular_data.multi_table.evaluation.transaction_evaluation import (
    account_bootstrap,
    standardized_ap,
)
from implementations.tabular_data.multi_table.evaluation.transaction_quality import (
    account_profiles,
    correlation_differences,
    distribution_differences,
    transaction_quality,
)
from implementations.tabular_data.multi_table.experiments import tuning


@pytest.fixture
def tables():
    """Small dataset with both event classes in every account."""
    trans = pd.DataFrame(
        {
            "trans_id": np.arange(80),
            "account_id": np.repeat(np.arange(20), 4),
            "trans_date": np.tile([0, 1, 2, 3], 20),
            "amount": np.tile([10, 20, 30, 40], 20),
            "balance": np.tile([20, 10, -5, 30], 20),
            **{name: np.tile([0.0, 1.0, 2.0, 0.0], 20) for name in ("trans_type", "operation", "k_symbol", "bank")},
        },
    )
    accounts = pd.DataFrame({"account_id": np.arange(21), "account_date": 10, "frequency": np.arange(21) % 3})
    return {"trans": trans, "account": accounts}


def test_category_codes_preserve_float_integer_missing_and_unknown():
    """Changing a CSV dtype must not change the category meanings."""
    reference = pd.Series([0.0, 1.0, np.nan])
    aligned_reference = align_categories(reference, reference)
    synthetic = pd.Series([0, 1.0, 0.8, 8, np.nan, np.inf])
    expected = ["0", "1", "1", "__unknown__", "__missing__", "__unknown__"]
    assert align_categories(synthetic, aligned_reference).tolist() == expected


def test_detector_uses_disjoint_accounts_and_excludes_target(tables):
    """Neither balance nor account IDs are classifier predictors."""
    evaluator = OverdraftTransferEvaluator(tables["trans"], random_forest_params={"n_estimators": 5, "n_jobs": 1})
    assert set(evaluator.real_train.account_id).isdisjoint(evaluator.real_test.account_id)
    assert "balance" not in FEATURES and "account_id" not in FEATURES
    before = evaluator.evaluate_synthetic(tables["trans"], name="before")
    probabilities = evaluator.synthetic_probabilities["before"].copy()
    changed = tables["trans"].copy()
    changed["balance"] *= -1
    evaluator.evaluate_synthetic(changed, name="after")
    np.testing.assert_array_equal(probabilities, evaluator.synthetic_probabilities["after"])
    assert before["overdrafts"] != evaluator.get_metrics().loc["after", "overdrafts"]


def test_single_class_is_undefined_and_invalid_balance_rejected(tables):
    """Missing positives must not become a perfect or silently zero AP score."""
    scores = classification_metrics(np.zeros(4), np.arange(4) / 4)
    assert np.isnan(scores["average_precision"]) and np.isnan(scores["roc_auc"])
    tables["trans"]["balance"] = tables["trans"].balance.astype(float)
    tables["trans"].loc[0, "balance"] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        OverdraftTransferEvaluator(tables["trans"])


def test_identical_data_zero_distances_and_epoch_alignment(tables):
    """Offset zero in transactions is Jan 16, after account day ten (Jan 11)."""
    summary, details = transaction_quality(tables, tables, {"trans": "930116", "account": "930101"})
    assert summary["transaction_ks"] == 0
    assert summary["transaction_tvd"] == 0
    assert summary["transaction_correlation_rmse"] == 0
    assert summary["synthetic_transaction_before_account_rate"] == 0
    assert summary["undefined_account_pairs"] > 0  # Constant opening dates.
    assert details["synthetic_account_profiles"].iloc[-1].transactions == 0
    changed = deepcopy(tables)
    changed["account"]["account_date"] = 100
    other, _ = transaction_quality(tables, changed, {"trans": "930116", "account": "930101"})
    assert other["synthetic_transaction_before_account_rate"] == 1


def test_invalid_values_orphans_and_duplicates_are_visible(tables):
    """Invalid values do not disappear into an apparently good aggregate."""
    changed = deepcopy(tables)
    changed["trans"].loc[0, "account_id"] = 999
    changed["trans"].loc[1, "trans_id"] = changed["trans"].loc[0, "trans_id"]
    changed["trans"].loc[2, "bank"] = 99
    changed["trans"].loc[3, "amount"] = np.nan
    summary, _ = transaction_quality(tables, changed)
    assert summary["transaction_fk_valid_rate"] == 79 / 80
    assert summary["transaction_pk_unique_rate"] == 79 / 80
    assert summary["max_unknown_category_rate"] == 1 / 80
    assert summary["max_invalid_rate"] == 1 / 80
    changed["account"].loc[1, "account_id"] = 0
    with pytest.raises(ValueError, match="unique"):
        account_profiles(changed["account"], changed["trans"])


def test_conditionals_find_damage_hidden_in_marginals(tables):
    """Swapping which transactions overdraw preserves balance marginals."""
    changed = deepcopy(tables)
    changed["trans"]["balance"] = np.roll(changed["trans"].balance, 1)
    summary, _ = transaction_quality(tables, changed)
    assert summary["transaction_ks"] == 0
    assert summary["overdraft_numerical_ks"] > 0
    assert summary["overdraft_categorical_tvd"] > 0


def test_empty_class_and_constant_correlations_remain_missing(tables):
    """Missing comparison support must be exposed to the reader."""
    distances = distribution_differences(tables["trans"].iloc[:0], tables["trans"], ["amount"], ["bank"])
    assert distances.distance.isna().all()
    frame = pd.DataFrame({"a": [1, 1], "b": [2, 3]})
    assert correlation_differences(frame, frame, ["a", "b"], []).absolute_error.isna().all()


def test_standardized_ap_removes_only_class_prior_difference():
    """Duplicating every negative alters raw AP but not prior-standardized AP."""
    y, p = np.array([0, 1, 0, 1]), np.array([0.9, 0.8, 0.2, 0.1])
    y2, p2 = np.r_[y, y[y == 0]], np.r_[p, p[y == 0]]
    assert average_precision_score(y, p) != average_precision_score(y2, p2)
    assert standardized_ap(y, p, 0.1) == pytest.approx(standardized_ap(y2, p2, 0.1))


def test_bootstrap_samples_accounts_with_multiplicity():
    """Compare against explicit resampling; never split an account's rows."""
    frame = pd.DataFrame({"account_id": [10, 10, 20, 20, 30, 30], "overdraft": [0, 1, 0, 1, 0, 1]})
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.4, 0.6])
    actual = account_bootstrap(frame, scores, 0.5, repetitions=40, seed=7)
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(40):
        selected = rng.integers(3, size=3)
        index = np.concatenate([np.array([2 * i, 2 * i + 1]) for i in selected])
        expected.append(average_precision_score(frame.overdraft.iloc[index], scores[index]))
    assert actual["average_precision_low"] == pytest.approx(np.quantile(expected, 0.025))
    assert actual["average_precision_high"] == pytest.approx(np.quantile(expected, 0.975))
    assert actual["bootstrap_valid"] == 40


@pytest.mark.parametrize("preset", tuning.PRESETS)
def test_presets_validate_and_preserve_base(preset):
    """Every advertised hyperparameter is an actual accepted config field."""
    base = tuning.base_config()
    original = deepcopy(base)
    parameter, values = tuning.PRESETS[preset]
    assert len(tuning.make_trials(base, parameter, values)) == len(values)
    assert base == original


def test_cache_dependencies_change_only_affected_stages():
    """Guidance reuses training; classifier tuning reuses diffusion."""
    base = tuning.base_config()
    original = tuning.stage_ids(base, "same data")
    for parameter, changed_keys in [
        ("sampling_config.classifier_scale", {"sample"}),
        ("classifier_config.lr", {"model", "sample"}),
        ("diffusion_config.lr", {"diffusion", "model", "sample"}),
        ("clustering_config.parent_scale", {"cluster", "diffusion", "model", "sample"}),
    ]:
        section, field = parameter.split(".")
        changed = tuning.make_trials(base, parameter, [base[section][field] * 2])[0]
        ids = tuning.stage_ids(changed, "same data")
        assert {k for k in original if original[k] != ids[k]} == changed_keys
    assert all(v != tuning.stage_ids(base, "changed data")[k] for k, v in original.items())


def test_configuration_rejects_ignored_keys_and_mismatched_splits():
    """Configuration mistakes must fail before training starts."""
    base = tuning.base_config()
    base["sampling_config"]["guidence"] = 1
    with pytest.raises(ValueError, match="Unknown"):
        tuning.validate_config(base)
    base = tuning.base_config()
    base["classifier_config"]["data_split_ratios"] = [0.8, 0.1, 0.1]
    with pytest.raises(ValueError, match="share"):
        tuning.validate_config(base)
    with pytest.raises(ValueError, match="positive"):
        tuning.make_trials(tuning.base_config(), "diffusion_config.lr", [-1])


def test_stage_reuses_complete_rejects_changed_and_failed(tmp_path):
    """A saved status alone cannot hide a changed artifact or failed training."""
    calls = []

    def write(folder):
        calls.append(1)
        path = folder / "artifact.txt"
        path.write_text("trained")
        return [path]

    folder = tuning._stage(tmp_path / "good", {"seed": 42}, write)
    tuning._stage(folder, {"seed": 42}, write)
    assert len(calls) == 1
    (folder / "artifact.txt").write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        tuning._stage(folder, {"seed": 42}, write)

    def fail(folder):
        raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        tuning._stage(tmp_path / "failed", {}, fail)
    with pytest.raises(ValueError, match="Incomplete"):
        tuning._stage(tmp_path / "failed", {}, write)


class DummyNetwork:
    """Picklable lightweight substitute used only to exercise orchestration."""

    def eval(self):
        """Record inference mode."""
        self.evaluation_mode = True


def test_runner_dispatch_reuse_and_actual_matched_keys(tmp_path, monkeypatch, tables):
    """Exercise the entire runner without training a real generator or using GPU."""
    from midst_toolkit.models.clavaddpm import clustering, data_loaders, synthesizer, train  # noqa: PLC0415

    calls = {"diffusion": 0, "classifier": 0, "sample": 0, "cluster": 0}
    relations = [(None, "account"), ("account", "trans")]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_meta.json").write_text("{}")

    def load(_):
        return {k: SimpleNamespace(data=v.copy(), domain={}) for k, v in tables.items()}, relations, {}

    def cluster(source, order, folder, config):
        calls["cluster"] += 1
        tuning._dump(folder / "cluster_ckpt.pkl", {"tables": source, "all_group_lengths_prob_dicts": {}})

    def diffuse(*args):
        calls["diffusion"] += 1
        assert args[-2].iterations == 0
        return SimpleNamespace(
            diffusion=DummyNetwork(), classifier=None, table_metadata=None, model_parameters=None, transformations=None,
        )

    def classifier(*args, **kwargs):
        calls["classifier"] += 1
        return DummyNetwork()

    def match(*args, **kwargs):
        return {k: v.copy() for k, v in tables.items()}

    def sample(source, order, folder, models, *args):
        calls["sample"] += 1
        assert all(m.diffusion.evaluation_mode for m in models.values())
        assert models[("account", "trans")].classifier.evaluation_mode
        final = synthesizer.clava_synthesizing_matching_process(None)
        # Toolkit cleanup may mutate its result: preserved keys must survive.
        final["trans"].drop(columns=["trans_id"], inplace=True)

    monkeypatch.setattr(data_loaders, "load_tables", load)
    monkeypatch.setattr(clustering, "clava_clustering", cluster)
    monkeypatch.setattr(train, "child_training", diffuse)
    monkeypatch.setattr(train, "train_classifier", classifier)
    monkeypatch.setattr(synthesizer, "clava_synthesizing", sample)
    monkeypatch.setattr(synthesizer, "clava_synthesizing_matching_process", match)
    base = tuning.base_config()
    trials = tuning.make_trials(base, "sampling_config.classifier_scale", [0, 1])
    trials += tuning.make_trials(base, "classifier_config.lr", [base["classifier_config"]["lr"] * 2])
    paths = tuning.run_trials(trials, data_dir, tmp_path / "outputs", device="cpu")
    assert calls == {"cluster": 1, "diffusion": 2, "classifier": 2, "sample": 3}
    assert "trans_id" in tuning._load(paths[0] / "synthetic_final_with_keys.pkl")["trans"]
    assert synthesizer.clava_synthesizing_matching_process is match
    tuning.run_trials(trials, data_dir, tmp_path / "outputs", device="cpu")
    assert calls == {"cluster": 1, "diffusion": 2, "classifier": 2, "sample": 3}
