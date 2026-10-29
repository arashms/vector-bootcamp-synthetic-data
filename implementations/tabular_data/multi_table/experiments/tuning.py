"""Small notebook helpers for controlled, cached ClavaDDPM experiments.

Planning never trains or samples. Execution is explicit. Each stage is isolated
under the experiment output directory; existing baseline checkpoints are read-only.
"""

# Lazy toolkit imports keep hash/plan inspection independent of model loading.
# ruff: noqa: PLC0415
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import pandas as pd
import yaml


MULTI_TABLE = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_ROOT = MULTI_TABLE / "results/experiments/transaction_tuning_from_scratch"
DEFAULT_REPORT_ROOT = MULTI_TABLE / "results/experiments/transaction_overdraft_from_scratch"
PRESETS = {
    "guidance": ("sampling_config.classifier_scale", [0.0, 0.5, 1.0, 2.0, 4.0]),
    "cluster_count": ("clustering_config.num_clusters", [10, 25, 50]),
    "parent_weight": ("clustering_config.parent_scale", [0.5, 1.0, 2.0]),
    "clustering_method": ("clustering_config.clustering_method", ["kmeans", "gmm", "kmeans_and_gmm"]),
    "diffusion_lr": ("diffusion_config.lr", [0.0002, 0.0006, 0.001]),
    "diffusion_steps": ("diffusion_config.iterations", [2500, 5000, 10000]),
    "diffusion_size": ("diffusion_config.d_layers", [[256, 512, 512, 256], [512, 1024, 1024, 1024, 1024, 512]]),
    "dropout": ("diffusion_config.dropout", [0.0, 0.1, 0.2]),
    "timesteps": ("diffusion_config.num_timesteps", [250, 500, 1000]),
    "diffusion_batch": ("diffusion_config.batch_size", [128, 256, 512]),
    "classifier_lr": ("classifier_config.lr", [0.00003, 0.0001, 0.0003]),
    "classifier_steps": ("classifier_config.iterations", [2500, 5000, 10000]),
    "classifier_size": ("classifier_config.d_layers", [[128, 256, 256, 128], [128, 256, 512, 1024, 512, 256, 128]]),
    "classifier_batch": ("classifier_config.batch_size", [128, 256, 512]),
    "classifier_time_embedding": ("classifier_config.dim_t", [64, 128, 256]),
}


def digest(value):
    """Hash a resolved configuration deterministically."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()[:20]


def file_hash(path):
    """Hash file contents, not timestamps."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def data_signature(data_dir):
    """Fingerprint tables, domains, and preprocessing metadata."""
    paths = sorted(p for p in Path(data_dir).iterdir() if p.suffix in {".csv", ".json"})
    if not paths or not (Path(data_dir) / "dataset_meta.json").exists():
        raise ValueError("Data directory must contain preprocessed tables and dataset_meta.json.")
    return {p.name: file_hash(p) for p in paths}


def backend_signature():
    """Record package versions and the source used by training and sampling."""
    import midst_toolkit

    root = Path(midst_toolkit.__file__).parent
    paths = sorted((root / "models/clavaddpm").glob("*.py")) + sorted((root / "common").glob("*.py"))
    return {
        "versions": {
            p: importlib.metadata.version(p) for p in ("midst-toolkit", "torch", "numpy", "pandas", "scikit-learn")
        },
        "source": digest({str(p.relative_to(root)): file_hash(p) for p in paths}),
        "runner": file_hash(__file__),
    }


def base_config(*, fresh_training=True):
    """Copy repository configs; align model preprocessing for NEW checkpoints."""
    config = yaml.safe_load((MULTI_TABLE / "training/config.yaml").read_text())
    config.update(yaml.safe_load((MULTI_TABLE / "synthesizing/config.yaml").read_text()))
    config.pop("random_seed", None)
    config["training_seed"] = 42
    config["sampling_seed"] = 42
    config["classifier_config"].setdefault("data_split_ratios", [0.7, 0.2, 0.1])
    if fresh_training:
        config["diffusion_config"]["data_split_ratios"] = [0.7, 0.2, 0.1]
    return validate_config(config, fresh_training=fresh_training)


def validate_config(config, *, fresh_training=True):  # noqa: PLR0912
    """Reject ignored keys, invalid ranges, and incompatible preprocessing."""
    import numpy as np
    from midst_toolkit.common.config import (
        ClavaDDPMClassifierConfig,
        ClavaDDPMClusteringConfig,
        ClavaDDPMDiffusionConfig,
        ClavaDDPMMatchingConfig,
        ClavaDDPMSamplingConfig,
    )

    result = deepcopy(config)
    classes = {
        "clustering_config": ClavaDDPMClusteringConfig,
        "diffusion_config": ClavaDDPMDiffusionConfig,
        "classifier_config": ClavaDDPMClassifierConfig,
        "sampling_config": ClavaDDPMSamplingConfig,
        "matching_config": ClavaDDPMMatchingConfig,
    }
    if set(result) != {*classes, "training_seed", "sampling_seed"}:
        raise ValueError("Expected model, clustering, sampling, matching configs and two seeds only.")
    for section, cls in classes.items():
        values = result[section]
        sample_scale = values.pop("sample_scale", 1.0) if section == "sampling_config" else None
        unknown = set(values) - set(cls.model_fields)
        if unknown:
            raise ValueError(f"Unknown settings in {section}: {sorted(unknown)}")
        result[section] = cls(**values).model_dump(mode="json")
        if sample_scale is not None:
            if not np.isfinite(sample_scale) or sample_scale <= 0:
                raise ValueError("sample_scale must be positive and finite.")
            result[section]["sample_scale"] = sample_scale
        for key, value in result[section].items():
            if key in {
                "iterations",
                "batch_size",
                "num_timesteps",
                "dim_t",
                "lr",
                "num_matching_clusters",
                "matching_batch_size",
            } and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{section}.{key} must be positive and finite.")
            if key in {"weight_decay", "parent_scale", "classifier_scale"} and (not np.isfinite(value) or value < 0):
                raise ValueError(f"{section}.{key} must be nonnegative and finite.")
            if key == "dropout" and not 0 <= value < 1:
                raise ValueError("dropout must be in [0, 1).")
            if key == "d_layers" and (not value or any(v <= 0 for v in value)):
                raise ValueError("Network widths must be positive.")
            if key == "num_clusters":
                counts = value.values() if isinstance(value, dict) else [value]
                if any(v <= 0 for v in counts):
                    raise ValueError("Cluster counts must be positive.")
            if key == "data_split_ratios" and (any(v <= 0 for v in value) or not np.isclose(sum(value), 1)):
                raise ValueError("Use three positive split fractions summing to one.")
    if (
        fresh_training
        and result["diffusion_config"]["data_split_ratios"] != result["classifier_config"]["data_split_ratios"]
    ):
        raise ValueError("Fresh diffusion/classifier training must share data_split_ratios.")
    for key in ("training_seed", "sampling_seed"):
        if not isinstance(result[key], int) or not 0 <= result[key] < 2**32:
            raise ValueError(f"{key} must be an integer in [0, 2**32).")
    return result


def make_trials(
    base, parameter, values, *, sampling_seeds=(42,), training_seeds=None, fresh_training=True, include_baseline=False,
):
    """Vary one parameter; optionally include its baseline for every seed pair."""
    section, field = parameter.split(".")
    if section not in base or field not in base[section]:
        raise ValueError(f"Unknown parameter: {parameter}")
    candidates = []
    for value in ([base[section][field]] if include_baseline else []) + list(values):
        if value not in candidates:
            candidates.append(deepcopy(value))
    trials = []
    for train_seed in training_seeds or (base["training_seed"],):
        for value in candidates:
            for sample_seed in sampling_seeds:
                config = deepcopy(base)
                config[section][field] = value
                config.update(training_seed=train_seed, sampling_seed=sample_seed)
                trials.append(validate_config(config, fresh_training=fresh_training))
    return trials


def stage_ids(config, context):
    """Identify each stage by exactly the upstream settings it depends on."""
    cluster = digest({"context": context, "clustering": config["clustering_config"], "seed": config["training_seed"]})
    diffusion = digest({"cluster": cluster, "diffusion": config["diffusion_config"]})
    model = digest({"diffusion": diffusion, "classifier": config["classifier_config"]})
    sample = digest(
        {
            "model": model,
            "sampling": config["sampling_config"],
            "matching": config["matching_config"],
            "seed": config["sampling_seed"],
        },
    )
    return {"cluster": cluster, "diffusion": diffusion, "model": model, "sample": sample}


def preview_trials(trials):
    """Logical reuse groups; no files, models, or GPU work are created."""
    rows = []
    for config in trials:
        rows.append(
            {
                **stage_ids(config, context="preview only"),
                "training_seed": config["training_seed"],
                "sampling_seed": config["sampling_seed"],
                "guidance": config["sampling_config"]["classifier_scale"],
                "clusters": config["clustering_config"]["num_clusters"],
                "config": config,
            },
        )
    return pd.DataFrame(rows)


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _stage(directory, manifest, action):
    """Reuse completed, verified artifacts; never mistake partial work for success."""
    directory = Path(directory)
    status_path = directory / "status.json"
    if directory.exists():
        if not status_path.exists():
            raise ValueError(f"Untracked stage directory: {directory}. Use a fresh output_root.")
        status = json.loads(status_path.read_text())
        if status["status"] != "complete" or json.loads((directory / "manifest.json").read_text()) != manifest:
            raise ValueError(f"Incomplete or incompatible stage: {directory}. Use a fresh output_root to retry.")
        if any(
            not (directory / name).is_file() or file_hash(directory / name) != checksum
            for name, checksum in status["artifacts"].items()
        ):
            raise ValueError(f"Changed or missing stage artifacts: {directory}")
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    _save_json(directory / "manifest.json", manifest)
    _save_json(status_path, {"status": "running"})
    start = perf_counter()
    try:
        files = action(directory)
        if not files or any(not p.is_file() for p in files):
            raise ValueError("Stage did not produce its expected artifacts.")
        _save_json(
            status_path,
            {
                "status": "complete",
                "seconds": perf_counter() - start,
                "artifacts": {str(p.relative_to(directory)): file_hash(p) for p in files},
            },
        )
    except BaseException as error:
        _save_json(status_path, {"status": "failed", "error": repr(error), "seconds": perf_counter() - start})
        raise
    return directory


def _load(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def completed_runs(data_dir, output_root, *, configs=None):
    """Discover completed samples for this dataset, optionally exact trial configs.

    Partial stages and other datasets are excluded. A claimed complete run with
    changed or missing artifacts fails explicitly. Discovery does not sample or
    train models, and includes compatible results from earlier runner versions.
    """
    signature = data_signature(data_dir)
    paths = []
    for status_path in sorted((Path(output_root) / "samples").glob("*/status.json")):
        status = json.loads(status_path.read_text())
        if status.get("status") != "complete":
            continue
        directory = status_path.parent
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("context", {}).get("data") != signature:
            continue
        if configs is not None and manifest.get("config") not in configs:
            continue
        artifacts = status.get("artifacts", {})
        expected = {"synthetic_final_with_keys.pkl", "config.yaml"}
        if not expected.issubset(artifacts) or any(
            not (directory / name).is_file() or file_hash(directory / name) != checksum
            for name, checksum in artifacts.items()
        ):
            raise ValueError(f"Changed or missing completed-run artifacts: {directory}")
        paths.append(directory)
    return paths


def _dump(path, value):
    with path.open("wb") as stream:
        pickle.dump(value, stream)


def run_trials(trials, data_dir, output_root, *, legacy_results=None, device="cuda"):
    """Execute explicitly chosen trials, returning saved run directories.

    legacy_results permits sampling-only changes to the original configuration.
    Fresh experiments cache clustering, diffusion, and classifiers independently.
    No work starts merely by importing this module or previewing a trial list.
    """
    import torch
    from midst_toolkit.models.clavaddpm.data_loaders import load_tables

    trials = [validate_config(c, fresh_training=legacy_results is None) for c in trials]
    data_dir, output_root = Path(data_dir).resolve(), Path(output_root).resolve()
    if output_root == MULTI_TABLE / "results" or output_root.is_relative_to(data_dir):
        raise ValueError("Use a dedicated experiment directory, separate from data and baseline results.")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Preview trials without launching; select a GPU kernel for execution.")
    context = {"data": data_signature(data_dir), "backend": backend_signature(), "device": str(device)}
    _, relations, _ = load_tables(data_dir)
    legacy = Path(legacy_results).resolve() if legacy_results else None
    if legacy:
        original = base_config(fresh_training=False)
        for config in trials:
            if any(
                config[k] != original[k]
                for k in ("clustering_config", "diffusion_config", "classifier_config", "training_seed")
            ):
                raise ValueError(
                    "Legacy checkpoints only support sampling/matching changes. Use fresh training for model changes.",
                )
        sources = [legacy / "cluster_ckpt.pkl"] + [legacy / "models" / f"{p}_{c}_ckpt.pkl" for p, c in relations]
        context["legacy_artifacts"] = {str(p): file_hash(p) for p in sources}
        stored = _load(legacy / "cluster_ckpt.pkl")
        for table, info in stored["tables"].items():
            pd.testing.assert_frame_equal(
                pd.read_csv(data_dir / f"{table}.csv"),
                info.data[info.original_column_names].reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
    return [
        _run_trial(
            config,
            context=context,
            data_dir=data_dir,
            output_root=output_root,
            relations=relations,
            legacy=legacy,
            device=device,
        )
        for config in trials
    ]


def _run_trial(config, *, context, data_dir, output_root, relations, legacy, device):  # noqa: PLR0915
    # Keep the four sequential stages together so the dependency order is visible.
    from midst_toolkit.common.config import (
        ClavaDDPMClassifierConfig,
        ClavaDDPMClusteringConfig,
        ClavaDDPMDiffusionConfig,
        ClavaDDPMMatchingConfig,
        ClavaDDPMSamplingConfig,
        GeneralConfig,
    )
    from midst_toolkit.common.random import set_all_random_seeds
    from midst_toolkit.models.clavaddpm import synthesizer
    from midst_toolkit.models.clavaddpm.clustering import clava_clustering
    from midst_toolkit.models.clavaddpm.data_loaders import NO_PARENT_COLUMN_NAME, load_tables
    from midst_toolkit.models.clavaddpm.train import child_training, get_df_without_id, train_classifier

    ids = stage_ids(config, context)
    common = {"context": context, "training_seed": config["training_seed"]}
    cluster_dir = legacy or output_root / "clustering" / ids["cluster"]
    if not legacy:

        def cluster_action(folder):
            set_all_random_seeds(config["training_seed"])
            tables, _, _ = load_tables(data_dir)
            clava_clustering(tables, relations, folder, ClavaDDPMClusteringConfig(**config["clustering_config"]))
            return [folder / "cluster_ckpt.pkl"]

        _stage(cluster_dir, {**common, "clustering": config["clustering_config"]}, cluster_action)
    clustered = _load(cluster_dir / "cluster_ckpt.pkl")
    tables = clustered["tables"]
    for parent, child in relations:
        if parent is None:
            tables[child].data[NO_PARENT_COLUMN_NAME] = list(range(len(tables[child].data)))
    diffusion_config = ClavaDDPMDiffusionConfig(**config["diffusion_config"])
    classifier_config = ClavaDDPMClassifierConfig(**config["classifier_config"])
    if legacy:
        model_dir = legacy / "models"
    else:
        diffusion_dir = output_root / "diffusion" / ids["diffusion"]

        def diffuse_action(folder):
            paths = []
            # Independent per-edge seeds make cache reuse irrelevant to randomness.
            for i, (parent, child) in enumerate(relations):
                set_all_random_seeds((config["training_seed"] + i) % 2**32)
                disabled = classifier_config.model_copy(update={"iterations": 0})
                artifact = child_training(
                    get_df_without_id(tables[child].data).copy(),
                    tables[child].domain,
                    parent,
                    child,
                    diffusion_config,
                    disabled,
                    device,
                )
                path = folder / f"{parent}_{child}_ckpt.pkl"
                _dump(path, artifact)
                paths.append(path)
                del artifact
            return paths

        _stage(
            diffusion_dir,
            {**common, "cluster": ids["cluster"], "diffusion": config["diffusion_config"]},
            diffuse_action,
        )
        model_dir = output_root / "models" / ids["model"]

        def classifier_action(folder):
            paths = []
            for i, (parent, child) in enumerate(relations):
                set_all_random_seeds((config["training_seed"] + 10000 + i) % 2**32)
                artifact = _load(diffusion_dir / f"{parent}_{child}_ckpt.pkl")
                if parent is not None:
                    artifact.classifier = train_classifier(
                        get_df_without_id(tables[child].data).copy(),
                        artifact.table_metadata,
                        artifact.model_parameters,
                        artifact.transformations,
                        diffusion_config,
                        classifier_config,
                        device=device,
                        cluster_col=f"{parent}_{child}_cluster",
                    )
                path = folder / f"{parent}_{child}_ckpt.pkl"
                _dump(path, artifact)
                paths.append(path)
                del artifact
            return paths

        _stage(
            model_dir,
            {**common, "diffusion": ids["diffusion"], "classifier": config["classifier_config"]},
            classifier_action,
        )
    run_dir = output_root / "samples" / ids["sample"]

    def sample_action(folder):
        models = {(p, c): _load(model_dir / f"{p}_{c}_ckpt.pkl") for p, c in relations}
        for artifact in models.values():
            artifact.diffusion.eval()
            if artifact.classifier is not None:
                artifact.classifier.eval()
        set_all_random_seeds(config["sampling_seed"])
        captured = {}
        original_match = synthesizer.clava_synthesizing_matching_process

        def capture_match(*args, **kwargs):
            final = original_match(*args, **kwargs)
            captured.update({table: frame.copy(deep=True) for table, frame in final.items()})
            return final

        general = GeneralConfig(
            data_dir=data_dir,
            test_data_dir=data_dir,
            exp_name=folder.name,
            workspace_dir=folder,
            sample_prefix="synthetic",
        )
        sampling = {k: v for k, v in config["sampling_config"].items() if k != "sample_scale"}
        # Capture the ACTUAL matched tables before toolkit cleanup drops keys.
        # This is scoped to one call and restored even on failure; run sequentially.
        with patch.object(synthesizer, "clava_synthesizing_matching_process", capture_match):
            synthesizer.clava_synthesizing(
                tables,
                relations,
                folder,
                models,
                general,
                ClavaDDPMSamplingConfig(**sampling),
                ClavaDDPMMatchingConfig(**config["matching_config"]),
                clustered["all_group_lengths_prob_dicts"],
                config["sampling_config"]["sample_scale"],
            )
        if set(captured) != set(tables):
            raise ValueError("Sampling did not return every table.")
        _dump(folder / "synthetic_final_with_keys.pkl", captured)
        (folder / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return [folder / "synthetic_final_with_keys.pkl", folder / "config.yaml"]

    _stage(run_dir, {"context": context, "config": config, "model_id": ids["model"]}, sample_action)
    return run_dir
