"""Checks for a zero-run start and matched controls without model dependencies."""

from copy import deepcopy

import pytest

from implementations.tabular_data.multi_table.experiments import tuning


def test_new_workflow_has_no_runs_and_creates_no_artifacts(tmp_path):
    """Discovering runs before the first experiment is an empty, read-only lookup."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_meta.json").write_text("{}")
    output_root = tmp_path / "never-run"
    assert tuning.completed_runs(data_dir, output_root) == []
    assert not output_root.exists()


@pytest.mark.parametrize("baseline,values", [(1.0, [0.5, 0.5]), ([128, 256], [[64, 128], [128, 256]])])
def test_baseline_control_is_included_once_for_each_seed_pair(monkeypatch, baseline, values):
    """Narrowing scalar or network candidates retains independent matched controls."""
    monkeypatch.setattr(tuning, "validate_config", lambda config, **kwargs: config)
    base = {"model": {"parameter": baseline}, "training_seed": 42, "sampling_seed": 42}
    original = deepcopy(base)
    candidates = deepcopy(values)
    trials = tuning.make_trials(
        base, "model.parameter", values, training_seeds=[42, 123], sampling_seeds=[42, 456],
        include_baseline=True,
    )
    for train_seed in (42, 123):
        for sample_seed in (42, 456):
            selected = [trial["model"]["parameter"] for trial in trials
                        if trial["training_seed"] == train_seed and trial["sampling_seed"] == sample_seed]
            assert len(selected) == 2
            assert selected.count(baseline) == 1
    assert base == original
    assert values == candidates
    trials[0]["model"]["parameter"] = "changed"
    assert base == original


def test_completed_run_discovery_filters_data_configs_and_checks_artifacts(tmp_path):
    """Notebook discovery excludes partial/unrelated data and detects corruption."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "dataset_meta.json").write_text("{}")
    root = tmp_path / "experiment"

    def save(folder):
        paths = [folder / "synthetic_final_with_keys.pkl", folder / "config.yaml"]
        for path in paths:
            path.write_text("test artifact")
        return paths

    config = {"parameter": 1}
    manifest = {"context": {"data": tuning.data_signature(data_dir)}, "config": config}
    complete = tuning._stage(root / "samples" / "complete", manifest, save)
    tuning._stage(root / "samples" / "other-data", {**manifest, "context": {"data": {}}}, save)
    partial = root / "samples" / "partial"
    partial.mkdir()
    (partial / "status.json").write_text('{"status": "failed"}')
    assert tuning.completed_runs(data_dir, root) == [complete]
    assert tuning.completed_runs(data_dir, root, configs=[config]) == [complete]
    assert tuning.completed_runs(data_dir, root, configs=[{"parameter": 2}]) == []
    assert tuning.completed_runs(data_dir, root, configs=[]) == []
    (complete / "config.yaml").write_text("changed")
    with pytest.raises(ValueError, match="Changed"):
        tuning.completed_runs(data_dir, root)

