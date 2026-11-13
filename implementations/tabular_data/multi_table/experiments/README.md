# Multi-table experiments (ClavaDDPM on Berka)

Tuning ClavaDDPM on a 500-account Berka cohort, judged by full multi-table quality
and by how well a real-trained overdraft detector transfers to synthetic data.
**Results: [docs/TUNING_SUMMARY.md](docs/TUNING_SUMMARY.md).**

## Layout

```text
experiments/
├── tuning.py             training/sampling/caching (do not edit: its hash keys the cache)
├── dataset_settings.py   DATA_DIR, account count, seed (restart kernels after editing)
├── run_sweep.py          terminal launcher for multi-GPU sweeps
├── analysis/             post-hoc scripts (overdraft episodes, train-on-synthetic test)
├── notebooks/            run order 00 → 01 → sweeps → 02
│   ├── sweeps/           one notebook per hyperparameter
│   └── archive/          historical notebook (original checkpoints, old workflow)
└── docs/                 summary, decision log, handoff notes
```

## Run order

1. `uv sync --locked --group tabular-data --inexact`, select the `.venv` kernel.
2. [00_prepare_account_sample.ipynb](notebooks/00_prepare_account_sample.ipynb):
   build the cohort (CPU, 500 accounts, seed 42).
3. [01_baseline.ipynb](notebooks/01_baseline.ipynb): train, generate, evaluate
   the reference (GPU).
4. Sweeps, one parameter at a time:
   [guidance_strength_tuning](notebooks/sweeps/guidance_strength_tuning.ipynb),
   [cluster_count](notebooks/sweeps/cluster_count.ipynb),
   [parent_weight](notebooks/sweeps/parent_weight.ipynb),
   [clustering_method](notebooks/sweeps/clustering_method.ipynb).
5. [02_all_experiments.ipynb](notebooks/02_all_experiments.ipynb): compare all
   completed runs with one detector (CPU).

Notebooks never start training until you set `RUN_NEW_TRIALS = True`.

## Several GPUs

Stages are file-locked and cached; workers reuse each other's stages. Pin GPUs with
`CUDA_VISIBLE_DEVICES` and keep `device="cuda"`. From the repository root:

```bash
M=implementations.tabular_data.multi_table.experiments.run_sweep
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m $M baseline
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m $M cluster_count parent_weight \
    --training-seeds 42 43 44 --shard 0 --num-shards 2   # same args per worker
```

`--dry-run` lists a worker's trials. A killed worker leaves its stage `running`;
move that stage directory aside before retrying.

## Artifacts

Runs: `../results/experiments/transaction_tuning_from_scratch/`.
Reports: `../results/experiments/transaction_overdraft_from_scratch/<fingerprint>/`.
Do not move result folders; report IDs include run paths.

## Docs

| File | Contents |
| --- | --- |
| [TUNING_SUMMARY.md](docs/TUNING_SUMMARY.md) | Manager-facing results, experiments 1–15. |
| [PROGRESS.md](docs/PROGRESS.md) | Full decision log, metric definitions, caveats. |
| [HANDOFF.md](docs/HANDOFF.md) | Current state and next steps. |

Tests (CPU): `.venv/bin/python -m pytest -q implementations/tabular_data/multi_table/tests/`
