r"""Run preset sweeps from a terminal, splitting trials across GPU workers.

Run from the repository root. Pin each worker to one GPU with CUDA_VISIBLE_DEVICES
(keep device "cuda": the device string is part of every cache fingerprint) and
pass identical experiment/seed arguments to every worker, for example:

    M=implementations.tabular_data.multi_table.experiments.run_sweep
    for i in 0 1 2; do
        gpu=$(echo 4 6 7 | cut -d' ' -f$((i + 1)))
        CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=16 nohup .venv/bin/python -m $M \
            cluster_count parent_weight --shard $i --num-shards 3 \
            > sweep_$i.log 2>&1 &
    done

Evaluate afterwards in the parameter notebooks (same seeds) or 02_all_experiments.ipynb.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from implementations.tabular_data.multi_table.experiments.dataset_settings import DATA_DIR
from implementations.tabular_data.multi_table.experiments.tuning import (
    ACCOUNT_FEATURES,
    DATE_REPRESENTATIONS,
    DEFAULT_EXPERIMENT_ROOT,
    PRESETS,
    base_config,
    completed_runs,
    make_trials,
    run_trials,
    stage_ids,
)


SECTIONS = ("clustering_config", "diffusion_config", "classifier_config", "sampling_config", "matching_config")


def plan(
    experiments, sampling_seeds, training_seeds, date_representation="fraction_of_window", account_features=("none",),
):
    """Deduplicate trials; as in the notebooks, presets include their baseline."""
    base = base_config(fresh_training=True)
    base["date_representation"] = date_representation
    trials = []
    for feature in account_features:
        for name in experiments:
            parameter, values = PRESETS.get(name, ("sampling_config.classifier_scale", []))
            for config in make_trials(
                {**base, "account_feature": feature}, parameter, values, sampling_seeds=sampling_seeds,
                training_seeds=training_seeds, include_baseline=True,
            ):
                if config not in trials:
                    trials.append(config)
    return base, trials


def shard(trials, index, count):
    """Keep trials sharing trained models on one worker; balance by trial count."""
    groups = {}
    for config in trials:
        groups.setdefault(stage_ids(config, "shard")["model"], []).append(config)
    workers = [[] for _ in range(count)]
    for group in sorted(groups.values(), key=len, reverse=True):
        min(workers, key=len).extend(group)
    return workers[index]


def describe(config, base):
    """Summarize settings that differ from the baseline, plus seeds."""
    changes = [
        f"{section}.{key}={value}"
        for section in SECTIONS
        for key, value in config[section].items()
        if value != base[section].get(key)
    ]
    seeds = f"training seed {config['training_seed']}, sampling seed {config['sampling_seed']}"
    if config["date_representation"] != "fraction_of_window":
        changes.append(f"dates={config['date_representation']}")
    if config["account_feature"] != "none":
        changes.append(f"account_feature={config['account_feature']}")
    return f"{', '.join(changes) or 'baseline'} ({seeds})"


def main(argv=None):
    """Run this worker's share of the sweeps; return nonzero if any trial failed."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiments", nargs="+", choices=["baseline", *PRESETS])
    parser.add_argument("--sampling-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--date-representation", choices=DATE_REPRESENTATIONS, default="fraction_of_window")
    parser.add_argument("--account-features", nargs="+", choices=ACCOUNT_FEATURES, default=["none"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="List this worker's trials without GPU work.")
    args = parser.parse_args(argv)
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num-shards).")

    base, trials = plan(
        args.experiments, args.sampling_seeds, args.training_seeds, args.date_representation, args.account_features,
    )
    if "baseline" not in args.experiments and not completed_runs(
        args.data_dir, args.experiment_root, configs=[base],
    ):
        parser.error("No completed baseline for this dataset. Run 'baseline' and inspect it before sweeps.")
    mine = shard(trials, args.shard, args.num_shards)
    print(f"Worker {args.shard + 1}/{args.num_shards}: {len(mine)} of {len(trials)} trials.", flush=True)

    failures = 0
    for number, config in enumerate(mine, 1):
        label = f"[{number}/{len(mine)}] {describe(config, base)}"
        if args.dry_run:
            print(label, flush=True)
            continue
        print(f"{label}: started", flush=True)
        try:
            (run_dir,) = run_trials([config], args.data_dir, args.experiment_root, device="cuda")
        except Exception as error:  # noqa: BLE001 - independent trials remain useful after one failure.
            failures += 1
            print(f"{label}: FAILED {error!r}", flush=True)
            continue
        print(f"{label}: complete {run_dir}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
