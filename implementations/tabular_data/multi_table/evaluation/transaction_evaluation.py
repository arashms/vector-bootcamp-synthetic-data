"""Notebook-facing evaluation of saved, matched multi-table experiments.

Fits one real-data detector for the entire comparison. Generated tables are read
from local trusted pickle files; evaluation never loads generator checkpoints.
"""

# Plotting is imported only when a report is saved.
# ruff: noqa: PLC0415
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from implementations.tabular_data.multi_table.evaluation.overdraft_transfer_evaluator import (
    DEFAULT_RANDOM_FOREST_PARAMS,
    OverdraftTransferEvaluator,
)
from implementations.tabular_data.multi_table.evaluation.transaction_quality import transaction_quality


def standardized_ap(labels, probabilities, prevalence: float) -> float:
    """AP after weighting both classes to the same reference event prevalence.

    This removes the class-prior difference, not differences in task difficulty.
    It is a diagnostic alongside raw AP, not a replacement for held-out utility.
    """
    labels = np.asarray(labels)
    observed = labels.mean() if len(labels) else np.nan
    if not 0 < observed < 1 or not 0 < prevalence < 1:
        return np.nan
    weights = np.where(labels == 1, prevalence / observed, (1 - prevalence) / (1 - observed))
    return float(average_precision_score(labels, probabilities, sample_weight=weights))


def account_bootstrap(frame, probabilities, reference_prevalence, *, repetitions=200, seed=42):
    """Percentile intervals from resampling complete accounts, with replacement.

    Predictions remain fixed. One-class replicates are excluded and counted.
    These intervals do not quantify variation from refitting or generation seeds.
    """
    if repetitions < 0:
        raise ValueError("Bootstrap repetitions must be nonnegative.")
    if frame.account_id.isna().any():
        raise ValueError("Account bootstrap requires nonmissing account IDs.")
    labels = frame.overdraft.to_numpy()
    probabilities = np.asarray(probabilities)
    groups = list(frame.reset_index(drop=True).groupby("account_id", sort=False).indices.values())
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        # Concatenation preserves multiplicity when an account is drawn twice.
        indices = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        y, p = labels[indices], probabilities[indices]
        if np.unique(y).size != 2:
            continue
        values.append([average_precision_score(y, p), roc_auc_score(y, p), standardized_ap(y, p, reference_prevalence)])
    result = {"bootstrap_repetitions": repetitions, "bootstrap_valid": len(values)}
    for i, name in enumerate(("average_precision", "roc_auc", "standardized_ap")):
        bounds = np.quantile(np.asarray(values)[:, i], [0.025, 0.975]) if len(values) >= 20 else [np.nan, np.nan]
        result[f"{name}_low"], result[f"{name}_high"] = map(float, bounds)
    return result


def _hash_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_json(path, value):
    # pandas serializes numpy scalars and undefined values as standard JSON null.
    Path(path).write_text(pd.Series(value, dtype=object).to_json(indent=2), encoding="utf-8")


def evaluate_saved_runs(
    data_dir: str | Path,
    run_dirs: list[Path],
    output_root: str | Path,
    *,
    bootstrap_repetitions: int = 200,
    seed: int = 42,
    threshold: float = 0.5,
    random_forest_params: dict | None = None,
):
    """Save comparison CSV, per-run diagnostics, predictions, plots, and manifest.

    Returns (comparison, report_directory). Identical evaluations share a report
    directory; changed data/code/settings create a new one. Partial evaluations
    can be rerun. Original trial artifacts are always read-only.
    """
    data_dir = Path(data_dir).resolve()
    run_dirs = [Path(p).resolve() for p in run_dirs]
    if not run_dirs or len({p.name for p in run_dirs}) != len(run_dirs) or any(p.name == "real_test" for p in run_dirs):
        raise ValueError("Provide at least one run; directory names must be unique.")
    files = [data_dir / f"{table}.csv" for table in ("account", "trans")]
    files += [data_dir / "preprocess_meta.json"]
    files += list(Path(__file__).parent.glob("*.py"))
    for directory in run_dirs:
        files += [directory / "synthetic_final_with_keys.pkl"]
        files += [p for p in (directory / "config.yaml", directory / "manifest.json") if p.exists()]
    manifest = {
        "files": {str(p): _hash_file(p) for p in sorted(files)},
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "threshold": threshold,
        "forest": {**DEFAULT_RANDOM_FOREST_PARAMS, **(random_forest_params or {})},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "scipy", "scikit-learn", "midst-toolkit", "syntheval")
        },
        "task": "current balance < 0; real-trained detector to synthetic",
        "fidelity_reference": "full generator training reference; not an outer holdout",
        "uncertainty": "account bootstrap conditional on one fitted detector and one saved generation",
    }
    identifier = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    report_dir = Path(output_root).resolve() / identifier
    if any(report_dir.is_relative_to(p) for p in run_dirs + [data_dir]):
        raise ValueError("Evaluation output must be separate from source data and saved trials.")
    report_dir.mkdir(parents=True, exist_ok=True)
    _write_json(report_dir / "manifest.json", manifest)
    _write_json(report_dir / "status.json", {"status": "running"})
    try:
        comparison = _evaluate(
            data_dir,
            run_dirs,
            report_dir,
            repetitions=bootstrap_repetitions,
            seed=seed,
            threshold=threshold,
            forest_params=random_forest_params,
        )
    except BaseException as error:
        _write_json(report_dir / "status.json", {"status": "failed", "error": repr(error)})
        raise
    _write_json(report_dir / "status.json", {"status": "complete"})
    return comparison, report_dir


def _evaluate(data_dir, run_dirs, report_dir, *, repetitions, seed, threshold, forest_params):
    real = {table: pd.read_csv(data_dir / f"{table}.csv") for table in ("account", "trans")}
    metadata = json.loads((data_dir / "preprocess_meta.json").read_text())
    evaluator = OverdraftTransferEvaluator(
        real["trans"], random_state=seed, threshold=threshold, random_forest_params=forest_params,
    )
    reference_prevalence = float(evaluator.real_test.overdraft.mean())
    split = pd.concat(
        [
            evaluator.real_train[["account_id"]].drop_duplicates().assign(split="train"),
            evaluator.real_test[["account_id"]].drop_duplicates().assign(split="test"),
        ],
    )
    split.to_csv(report_dir / "real_account_split.csv", index=False)
    _write_json(
        report_dir / "real_split_counts.json",
        {
            label: {
                "transactions": len(frame),
                "accounts": frame.account_id.nunique(),
                "overdrafts": int(frame.overdraft.sum()),
            }
            for label, frame in (("train", evaluator.real_train), ("test", evaluator.real_test))
        },
    )
    rows, curves = [], []

    def record(name, frame, probabilities, metrics):
        metrics = dict(metrics)
        metrics["dataset"] = name
        metrics["accounts"] = frame.account_id.nunique()
        metrics["standardized_ap"] = standardized_ap(frame.overdraft, probabilities, reference_prevalence)
        metrics.update(
            account_bootstrap(frame, probabilities, reference_prevalence, repetitions=repetitions, seed=seed),
        )
        predictions = frame[["account_id", "overdraft"]].copy()
        predictions["probability"] = probabilities
        folder = report_dir / name
        folder.mkdir(exist_ok=True)
        predictions.to_csv(folder / "predictions.csv", index=False)
        precision, recall, _ = precision_recall_curve(frame.overdraft, probabilities)
        curves.append((name, recall, precision))
        rows.append(metrics)
        return folder, metrics

    record("real_test", evaluator.real_test, evaluator.real_test_probabilities, evaluator.real_test_metrics)
    _write_json(report_dir / "real_test" / "metrics.json", rows[0])
    for directory in run_dirs:
        print(f"Evaluating {directory.name}", flush=True)
        with (directory / "synthetic_final_with_keys.pkl").open("rb") as stream:
            synthetic = pickle.load(stream)
        config_path = directory / "config.yaml"
        config = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
        scores = evaluator.evaluate_synthetic(synthetic["trans"], name=directory.name, config=config)
        folder, metrics = record(
            directory.name,
            evaluator.synthetic_data[directory.name],
            evaluator.synthetic_probabilities[directory.name],
            scores,
        )
        quality, details = transaction_quality(real, synthetic, metadata.get("date_epochs_yymmdd"))
        metrics.update(quality)
        metrics["ap_gap_from_real_test"] = (
            metrics["average_precision"] - evaluator.real_test_metrics["average_precision"]
        )
        metrics["prevalence_gap_from_full_real"] = (
            metrics["overdraft_prevalence"] - evaluator.real_data.overdraft.mean()
        )
        metrics["valid_structure"] = bool(
            quality["transaction_fk_valid_rate"] == 1 and quality["transaction_pk_unique_rate"] == 1,
        )
        for name, detail in details.items():
            if name == "overdraft_by_category":
                codebooks = metadata.get("tables", {}).get("trans", {}).get("label_classes", {})

                def decode(row, codebooks=codebooks):
                    labels = codebooks.get(row["column"], [])
                    return (
                        labels[int(row.category)]
                        if str(row.category).isdigit() and int(row.category) < len(labels)
                        else row.category
                    )

                detail["category_label"] = detail.apply(decode, axis=1)
            detail.to_csv(folder / f"{name}.csv", index=False)
        _write_json(folder / "metrics.json", metrics)
    comparison = pd.DataFrame(rows).set_index("dataset")
    comparison.to_csv(report_dir / "comparison.csv")
    plot_comparison(comparison, curves, report_dir)
    return comparison


def plot_comparison(comparison, curves, report_dir):
    """Compact uncertainty, quality, and precision/recall figures for notebooks."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(
        axes,
        ("average_precision", "standardized_ap"),
        ("Raw average precision", "AP at the real test prevalence"),
        strict=True,
    ):
        for y, (_, row) in enumerate(comparison.iterrows()):
            ax.plot([row[f"{metric}_low"], row[f"{metric}_high"]], [y, y], color="0.4")
            ax.scatter(row[metric], y, color="#7b2cbf")
        ax.set_yticks(range(len(comparison)), comparison.index)
        ax.set_title(title + " (95% account bootstrap)")
        ax.set_xlim(left=0)
    fig.savefig(report_dir / "transfer_intervals.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name, recall, precision in curves:
        ax.step(recall, precision, where="post", label=name, alpha=0.8)
    ax.set(xlabel="Recall", ylabel="Precision", title="Real-trained detector: raw precision–recall curves")
    ax.legend(fontsize=8)
    fig.savefig(report_dir / "precision_recall.png", dpi=150)
    plt.close(fig)
    columns = [
        "transaction_ks",
        "transaction_tvd",
        "overdraft_numerical_ks",
        "transactions_per_account_ks",
        "account_transaction_correlation_rmse",
    ]
    ax = comparison.drop(index="real_test")[columns].plot.bar(figsize=(12, 5), rot=15)
    ax.set(ylabel="Distance (lower is closer)", title="Separate fidelity diagnostics; no combined score")
    ax.figure.tight_layout()
    ax.figure.savefig(report_dir / "quality_comparison.png", dpi=150)
    plt.close(ax.figure)
