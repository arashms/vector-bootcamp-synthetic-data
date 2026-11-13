# Handoff — multi-table transaction/overdraft experiments (updated 2026-11-12 16:56)

## Prompt to paste into a new agent session

```text
You are resuming work on the multi-table ClavaDDPM experiments in this repo:
implementations/tabular_data/multi_table/experiments/

Read these first, in order:
1. implementations/tabular_data/multi_table/experiments/docs/HANDOFF.md  (this file: state + plan)
2. implementations/tabular_data/multi_table/experiments/docs/PROGRESS.md (full decision record)
3. implementations/tabular_data/multi_table/experiments/tuning.py   (runner, matcher, caching)

Rules:
- I (the user) launch every GPU job and run notebooks myself. Do not start training,
  generation, or notebook execution. Give me exact commands instead. CPU-only checks,
  tests, and reading saved artifacts are fine.
- Use the i-have-adhd skill (.agents/skills/i-have-adhd/SKILL.md) for every reply:
  next action first, numbered steps, restate where we are, one concrete next action at the end.
- Do one change at a time and measure it over 3 training seeds, paired by seed.
- Record results and decisions in PROGRESS.md, in the last (conclusion) cell of
  experiments/notebooks/02_all_experiments.ipynb (always rewrite the full cell), and in
  experiments/docs/TUNING_SUMMARY.md (manager-facing, add a new numbered experiment).
- Max GPU packing: 5 GPUs (1-5; GPU 0 belongs to another user), ~6 jobs per GPU.

Start with "Where we are" and "Next steps" in HANDOFF.md. Steps 9–10 are done
(final config confirmed). Then tell me the single next action.
```

## Where we are

| Step | Status |
| --- | --- |
| Raw Berka data downloaded; 500-account cohort (seed 42) prepared | Done |
| Baseline on the H200 node | Done (5 min 27 s per run) |
| Sweeps: guidance, cluster count, parent weight, clustering method × 3 sampling seeds (33 runs) | Done; conclusions in each notebook's section 5 |
| Training-seed check: baseline, parent scale, clustering method × training seeds 42/43/44 | Done; recorded |
| Multi-parent matching fix (code + CPU replay + tests) | Done |
| Matching-fix baseline × training seeds 42/43/44 (steps 1–2) | Done; new baseline runs `a406d8…`, `0e0e72…`, `8a2b1c…` |
| Date representation fix (step 3b) | Done; `fraction_of_window` adopted (before-opening now 0%) |
| Step 4: `ever_overdrawn` account feature | Rejected (keep `account_feature: none`) |
| Running-balance post-processing (CPU) | Rejected |
| Step 5: clustering method | Keep `kmeans_and_gmm` (report `a7b223c72a6c677d`) |
| Step 6: diffusion iterations | **Adopted 10000** (report `edb433b7162d5fa9`) |
| Step 7: diffusion iterations 20000 | Rejected, keep 10000 (report `60063072d89dd766`) |
| Step 8: diffusion presets | **Adopted diffusion batch_size 512** (report `43e7f4d2b0d8d13d`); dropout, lr, size, batch 128, timesteps rejected or noise |
| Step 9: classifier sweep | Done: keep defaults, all within noise (report `0d7c36bb65b1bacc`) |
| Step 10: final config × 5 seeds (42–46) | Done: ROC-AUC .757 ± .042, AP .0051 ± .0016, overdrawn 35.5% (report `c480bd90c664b936`) |
| Overdraft diagnosis + design D (CPU) | Done: gap = row profile + episode structure; D is circular, diagnostic only (`overdraft_episodes.py`, report `overdraft_episodes/reports/32066784546dbc9d`) |
| Reverse test (train synthetic → test real, CPU) | Done: final generator ROC-AUC .887 (real .872), AP .085 (real .242); `overdraft_episodes/tstr/tstr.csv` |

Current `base_config()` (fresh training): `fraction_of_window` dates,
`kmeans_and_gmm`, diffusion `iterations=10000`, diffusion `batch_size=512`.
Current best (step 8 batch 512): detector ROC-AUC .770/.691/.806, standardized AP
.0051/.0036/.0074 (real-test ROC-AUC 0.872, AP 0.242). Overdrawn accounts still
~30–40% vs 5.6% real.

Nothing is committed. All changes are uncommitted on branch `dev/arash`;
`TUNING_SUMMARY.md` is untracked (the user undid an earlier commit/push).

## Key findings so far

1. **No hyperparameter fixes the two main failures.**
   - Accounts ever overdrawn: 31–66% synthetic vs **5.6%** real.
   - Transactions dated before account opening: 14–26% synthetic vs **0%** real.
2. **Transfer gap is about 40×.** The real-trained overdraft detector gets
   standardized AP 0.0025–0.006 on synthetic data vs **0.2422** on held-out real
   accounts (real ROC-AUC 0.8722).
3. **Retraining noise dominates.** Baseline ROC-AUC across training seeds 42/43/44
   is 0.769 / 0.614 / 0.653 (sd 0.081), vs ±0.007 across sampling seeds. Compare
   any setting that retrains over 3 training seeds, paired by seed.
4. **Toolkit multi-parent matching was broken.** `handle_multi_parent` keeps
   whichever `disp` version is smaller.
   - That caused orphan cards (193 across 43 runs).
   - It also left clients without a disposition (1,595) and accounts without one (2,039).
   - Cards could silently attach to the wrong disposition.
5. **Only repeatable tuning signal:** `kmeans` clustering lowers pre-opening
   transactions in all 5 seed pairs tested (about 15% vs 19–20%), at some ROC-AUC cost.
   Everything else is within noise or rejected. See the PROGRESS.md table.

## Code changes (earlier sessions; later steps are in PROGRESS.md)

| File | Change |
| --- | --- |
| `experiments/tuning.py` | Sampling batch 200 → 8192 in `base_config` (rows are sampled independently, so this only changes speed and RNG order). |
| `experiments/tuning.py` | `fcntl` lock per stage in `_stage`, so parallel workers wait and reuse instead of crashing. |
| `experiments/tuning.py` | `match_multi_parent` replaces toolkit `handle_multi_parent` during sampling. It keeps the client-side `disp` (the one cards came from), covers every account first, and uses each account-side slot at most once. |
| `experiments/tuning.py` | `_enforce_keys` drops clients and cards attached to unmatched dispositions and counts the drops in each run's `matching_report.json`. |
| `experiments/tuning.py` | `completed_runs(..., current_runner_only=True)` ignores runs from older runner code, so pre-fix and post-fix runs never mix. |
| `experiments/run_sweep.py` | New terminal launcher: presets or `baseline`, `--training-seeds`, `--sampling-seeds`, `--shard/--num-shards`, `--dry-run`. |
| `tests/` | 46 tests pass (new: matcher test, old-runner filter). |
| Notebooks | Section 5 conclusions written for guidance, cluster count, parent weight, clustering method. `01_baseline.ipynb` sections 5–6 updated. |

**Matching-fix replay** on 43 saved pre-matching tables (CPU only):

| | Toolkit | Fixed |
| --- | ---: | ---: |
| Orphan cards | 193 | 0 |
| Clients without a disposition | 1,595 | 0 (dropped and counted) |
| Accounts without a disposition | 2,039 | 794 |
| Max dispositions per account (real 2) | 2 | 2 |

The remaining 794 come only from runs with fewer synthetic clients than accounts.
Client and account counts are generated independently from district. Fixing that is
separate work.

## Next steps

### 0. Steps 9–10 DONE (2026-11-13). Tuning exhausted; next is structural work on overdraft concentration (section 2).

### 1. Step 9 — classifier sweep (done; procedure kept as template)

Launched by `/tmp/step9.sh` (30 shards, presets `baseline classifier_lr
classifier_steps classifier_size classifier_batch classifier_time_embedding`,
training seeds 42/43/44). Phase 1: shards 0–2 (baselines) on GPUs 1–3; phase 2:
shards 3–29 on GPU `1 + s % 5`. Logs `step9_classifier_<s>.log` in the repo root;
`step9_done.flag` appears when all finish. At 16:56: 30 logs, 3 complete, 0 failed.

Trials (× 3 seeds): baseline; classifier lr 3e-5 / 3e-4; classifier iterations
2500 / 10000; d_layers [128,256,256,128]; classifier batch 128 / 512; dim_t 64 / 256.

Status check (CPU):

```bash
cd ~/repo/vector-bootcamp-synthetic-data; date; ls step9_done.flag 2>/dev/null
echo complete: $(grep -l "): complete" step9_classifier_*.log | wc -l)   # want 30
echo failed: $(grep -lE "FAILED|Traceback" step9_classifier_*.log | tr '\n' ' ')
echo running: $(ps -eo args | grep -c "[r]un_sweep")
```

When done:
1. User runs `02_all_experiments.ipynb` (CPU); note the new report hash.
2. Agent reads `results/experiments/transaction_overdraft_from_scratch/<hash>/comparison_with_settings.csv`.
   Label settings with `classifier_config.*` **plus** `diffusion_config.iterations`
   and `diffusion_config.batch_size`; the notebook also picks up older runs
   (e.g. 20000 iterations, batch 256) that must not be mislabelled as baseline.
3. Pivot setting × training_seed; count seeds better than baseline (ROC-AUC,
   standardized AP, overdrawn accounts, transaction TVD, all-table KS) and means.
4. Adopt only 3/3 wins. Edit `base_config()` in `tuning.py` if adopting (this
   changes the runner hash, so everything retrains).
5. Record: PROGRESS.md "Step 9" section, notebook last cell (draft text kept in
   `/tmp/cell14.md`; may be gone after reboot — then read the cell), TUNING_SUMMARY.md
   experiment 11.

### 2. Later candidates

- Diffusion batch 1024 (512 won over 256).
- Confirm the final config over 5 training seeds.
- Main failure still open: overdrafts spread over too many accounts.

## Launch template (user runs; max packing)

```bash
cd ~/repo/vector-bootcamp-synthetic-data
.venv/bin/python -m implementations.tabular_data.multi_table.experiments.run_sweep <presets> \
  --training-seeds 42 43 44 --dry-run      # CPU: list trials, pick --num-shards = trial count
# Run baselines first (shared stages), then variants; one shard per job, GPU 1 + s % 5:
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  setsid nohup .venv/bin/python -m implementations.tabular_data.multi_table.experiments.run_sweep \
  <presets> --training-seeds 42 43 44 --shard $s --num-shards $N > stepX_$s.log 2>&1 < /dev/null &
```

A 10000-iteration run takes ~10–15 min; ~6 jobs per H200 fit (<1 GB each).

## Historical steps (done; kept for reference)

### 1. Run the baseline with the matching fix only (user, GPU)

```bash
M=implementations.tabular_data.multi_table.experiments.run_sweep
GPUS=(2 6 7)   # check `nvidia-smi` first; the node is shared
for i in 0 1 2; do
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPUS[$i]} OMP_NUM_THREADS=16 nohup .venv/bin/python -m $M \
    baseline --training-seeds 42 43 44 --sampling-seeds 42 \
    --shard $i --num-shards 3 > fix_$i.log 2>&1 &
done
watch -n 30 'grep -c ": complete" fix_*.log'   # done when each shows 1
```

Then run `02_all_experiments.ipynb` (CPU). Because of the runner filter, it should
find exactly these 3 runs.

### 2. Compare with the pre-fix baseline (agent, CPU)

Compare paired by training seed with the pre-fix runs. Those are excluded by
default; use `completed_runs(..., current_runner_only=False)` or read report
`b52d0d6409bd1389` directly.

| Pre-fix run | Training seed |
| --- | --- |
| `e2cf71fc9b2cb2bdcc74` | 42 |
| `fa627eaf5227b73e6be4` | 43 |
| `524edfcd0e6f14a9c5b2` | 44 |

Check:
- Orphan cards are 0.
- Accounts and clients without a disposition, and the counts in `matching_report.json`.
- Client and card counts (real 596 / 90).
- `client→disp`, `account→disp` and `disp→card` cardinality, and multi-parent correlation.
- That transfer and transaction metrics did not move beyond training noise.

Record the results in PROGRESS.md and `01_baseline.ipynb`.

### 3. Store dates relative to account opening (agent implements)

**Goal:** remove transactions (and loans/cards) dated before account opening. Real 0%;
currently about 14–26% synthetic.

**Why:**
- Child rows are generated using only the parent's cluster label, not the parent's actual date.
- Dates are absolute day offsets from per-table epochs in
  `preprocess_meta.json → date_epochs_yymmdd`: account 930103, trans 930103,
  loan 930803, card 940405. Always read the epochs; don't assume they're equal.

**Recommended design:** keep the processed CSVs absolute, so evaluation and the
data fingerprint stay unchanged, and transform inside the runner.
1. Before clustering, convert `trans.trans_date` and `loan.loan_date` to days
   since the parent account opened, after aligning epochs. Card `issued` should be
   relative to its account via disp → account. Verify in real data that all
   offsets are ≥ 0.
2. After matching (in `capture_match`, before `_enforce_keys`/capture), decode with
   the synthetic parent's `account_date` and clip negative offsets to 0. Cards
   only get an account after matching, so decode them after the matcher.
3. Add an explicit config switch, e.g. a top-level
   `date_representation: absolute | relative_to_account`, so cache IDs differ.
   `validate_config` currently rejects unknown top-level keys, so update it and
   `stage_ids`/`make_trials` handling.
4. Tests: round-trip encode/decode on real data, no pre-opening dates after decode,
   and stage IDs change with the switch.

**Alternative:** a new processed dataset with relative dates. That changes the
data signature and needs evaluator decoding. Not recommended.

**Measure:** baseline × training seeds 42/43/44 with the switch on, paired against
step 1's runs.

### 4. Overdraft concentration (after step 3)

Synthetic overdrafts are spread over about 31–66% of accounts vs 5.6% real,
because balance is generated row by row.

**Untested idea:** add an account-level feature (e.g. minimum balance or an
overdraft-prone flag) to the account table, so clustering groups overdraft-prone
accounts and guidance keeps their transactions together. Check that the detector's
features (`trans_date`, `amount`, `trans_type`, `operation`, `k_symbol`, `bank`)
cannot leak it.

### 5. Then reconsider tuning

Re-test `kmeans` against the new baseline over 3 training seeds. Diffusion and
classifier sweeps stay deferred until steps 3–4 move the main failures.

## Environment and conventions

- **Node:** shared, 8× H200; other users load some GPUs. Pick idle GPUs from
  `nvidia-smi`. Always pin with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=N`
  and keep `device="cuda"`. The device string is part of every cache fingerprint.
- **Python:** `.venv/bin/python` from the repo root. The tool shell may drop `cd`;
  use absolute paths or `PYTHONPATH=$PWD`.
- **Tests:** `CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest -q implementations/tabular_data/multi_table/tests/` (54 pass)
- **Lint:** `.venv/bin/ruff check implementations/tabular_data/multi_table/` (docstring/comment lines ≤ 88 chars).
- **Any edit to `tuning.py` changes the runner fingerprint.** All stages retrain,
  and older runs drop out of discovery. Batch edits before a GPU campaign.
- **Timings:** a full baseline run takes about 5.5 min (training about 4.5 min, generation 46 s).
  Generation-only runs take about 1 min.
- **Evaluation** is CPU only. Each notebook writes to its own report folder named by
  a hash of its run set. Notebooks can run in parallel, but never the same one twice.
- **Killed workers** leave a stage `running`. Move that one stage directory aside
  before retrying; don't start a new root.

## Key paths

| What | Path |
| --- | --- |
| Processed cohort | `implementations/tabular_data/multi_table/data/berka_accounts_500_seed42/` |
| Stage caches and samples | `implementations/tabular_data/multi_table/results/experiments/transaction_tuning_from_scratch/` |
| Evaluation reports | `implementations/tabular_data/multi_table/results/experiments/transaction_overdraft_from_scratch/<hash>/` |
| Sweep reports | guidance `be458f4c5ed13f61`, cluster count `6280291d14dc3a69`, parent weight `fef27de26a023405`, clustering method `17381feb1e1e40ef` |
| Training-seed reports | parent weight `b52d0d6409bd1389`, clustering method `4a98a4d7b7ff8dde` |
| Preprocessing | `data_preprocessing/pre_process_berka_all_tabels.py`, `data_preprocessing/account_sampling.py` |
| Evaluation code | `evaluation/transaction_evaluation.py`, `transaction_quality.py`, `multi_table_quality.py`, `overdraft_transfer_evaluator.py` |
| Toolkit source | `.venv/lib64/python3.12/site-packages/midst_toolkit/models/clavaddpm/` (`synthesizer.py`: sampling and matching) |

## Open question for the user

The processed cohort folder, the root `data/` folder, and the `sweep_*.log` /
`seeds_*.log` / `fix_*.log` files are untracked and not git-ignored. Add them to
`.gitignore` before committing?
