# Transaction and overdraft tuning — progress and decisions

Updated: 2026-11-12. **Moved to the H200 node. Cohort, baseline, and four sweeps
(guidance, cluster count, parent weight, clustering method; 33 runs × 3 sampling
seeds) are complete and evaluated. The training-seed check is complete: retraining
moves results far more than any setting, and multi-parent matching is broken.
Hyperparameter sweeps are paused; next are the matching and date-representation fixes.**
The user executes notebooks. The agent inspected completed artifacts and updated
documentation without launching training, generation, or notebook execution.

## Approved extension — baseline and first guidance sweep complete

The user approved the plan on 2026-10-29: **500 accounts, seed 42**, complete raw
transaction histories, all eight linked tables, and **real→synthetic overdraft
transfer plus full-table quality**. The user remains responsible for executing
notebooks. Existing datasets, trained models, and synthesized artifacts must remain.

- [x] Add reusable account-cohort selection and a preparation notebook with editable
  raw/output dataset paths, account count, and seed. Preserve existing data.
- [x] Extract full-table/relationship metrics, excluding identifiers, and add the
  client/account relationship through `disp` as a multi-parent diagnostic.
- [x] Connect both metric groups to baseline, parameter, and final notebooks;
  fingerprint every reference table/domain and reject mismatched trial datasets.
- [x] Update instructions for a different dataset path and fresh matching baseline.
- [x] Run focused CPU tests and static notebook checks; record results here.

## Current objective and run order

Everything from 2026-10-29 was regenerated on the H200 node on 2026-11-12; those
older artifacts are not on this machine. Sweeps showed that no tried setting fixes
overdraft concentration or chronology, so tuning is paused in favour of a
data-representation change.

1. **Completed:** raw data download and `00_prepare_account_sample.ipynb`; same
   500-account cohort (real-test detector scores identical to 2026-10-29).
2. **Completed:** baseline, training/sampling seeds 42/42 (5 min 27 s of stage time).
3. **Completed:** guidance, cluster count, parent weight, and clustering method
   sweeps × sampling seeds 42/43/44 (33 runs); conclusions in each notebook's section 5.
4. **Completed:** training-seed check (seeds 42/43/44) for baseline, parent scale,
   and clustering method.
5. **Next:** fix orphan cards, store dates as days since account opening, then a
   new baseline with 3 training seeds.
6. **Deferred:** diffusion and classifier sweeps (20 runs).

See `README.md` for the notebook index and detailed handoff.

## H200 node rerun — 2026-11-12

### Environment and code changes

- Shared 8× H200 node; used GPUs 2, 4, 6, 7 via `CUDA_VISIBLE_DEVICES` with
  `CUDA_DEVICE_ORDER=PCI_BUS_ID`. The device string stays `cuda` because it is part
  of every stage fingerprint.
- `tuning.py`: sampling batch 200 → 8192 (rows are sampled independently with the
  classifier in eval mode); per-stage `fcntl` lock so concurrent workers wait for
  and reuse a stage. The runner fingerprint changed, so 2026-10-29 stages would not
  be reused.
- New `run_sweep.py` terminal launcher: shards trials across GPU workers, keeps
  trials sharing a model set on one worker, and requires a completed baseline.
- 44 tests pass; Ruff clean.

### Baseline stage times (training/sampling seed 42/42)

| Stage | 2026-10-29 (batch 200) | 2026-11-12 (H200, batch 8192) |
| --- | ---: | ---: |
| Clustering | 20.0 s | 12 s |
| Diffusion | 226.9 s | 104 s |
| Guidance classifier | 352.0 s | 165 s |
| Generation + matching | 1,449.6 s | 46 s |
| **Total** | **34 min 08 s** | **5 min 27 s** |

### Sweep results

Mean over sampling seeds 42/43/44; one training seed (42) per setting. Baseline is
guidance 1, 50 clusters, parent scale 1, `kmeans_and_gmm`. Real reference: test AP
0.2422, ROC-AUC 0.8722; 5.6% of accounts ever overdrawn; 0% of transactions before
account opening.

| Setting | Standardized AP ↑ | ROC-AUC ↑ | Accounts ever overdrawn | Before opening | All-table numerical KS ↓ | Min FK validity | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Baseline** | 0.0054 | 0.763 | 62.6% | 18.9% | 0.104 | 1.000 | Keep |
| Guidance 0 | 0.0054 | 0.769 | 65.7% | 21.3% | 0.109 | 1.000 | Equivalent |
| Guidance 0.5 | 0.0060 | 0.771 | 64.4% | 19.5% | 0.112 | 1.000 | Equivalent |
| Guidance 2 | 0.0049 | 0.737 | 62.4% | 18.6% | 0.126 | 1.000 | Reject |
| Guidance 4 | 0.0044 | 0.716 | 61.8% | 21.2% | 0.148 | 1.000 | Reject |
| 10 clusters | 0.0050 | 0.760 | 63.0% | 24.2% | 0.104 | 0.950 | No consistent gain; not retested |
| 25 clusters | 0.0052 | 0.754 | 58.8% | 22.9% | 0.102 | 0.976 | No consistent gain; +28% transactions |
| Parent scale 0.5 | 0.0048 | 0.749 | 65.4% | 21.5% | 0.103 | 1.000 | Reject: worse cardinality in every training seed |
| Parent scale 2.0 | 0.0052 | 0.776 | 65.0% | 23.1% | 0.094 | 1.000 | Reject: gains did not hold across training seeds |
| `gmm` | 0.0054 | 0.772 | 65.0% | 19.4% | 0.089 | 0.978 | Small ROC-AUC gain in every training seed; below retraining noise |
| `kmeans` | 0.0048 | 0.743 | 64.5% | 15.3% | 0.092 | 0.982 | Fewer pre-opening transactions in every seed; revisit after date fix |

Reports: [guidance](../../results/experiments/transaction_overdraft_from_scratch/be458f4c5ed13f61/comparison.csv),
[cluster count](../../results/experiments/transaction_overdraft_from_scratch/6280291d14dc3a69/comparison.csv),
[parent weight](../../results/experiments/transaction_overdraft_from_scratch/fef27de26a023405/comparison.csv),
[clustering method](../../results/experiments/transaction_overdraft_from_scratch/17381feb1e1e40ef/comparison.csv).
Per-setting detail and seed spread are in each notebook's section 5.

### Findings

1. **No setting fixes the main failures.** Accounts ever overdrawn stay at 59–66%
   (real 5.6%); transactions before opening stay at 15–24% (real 0%).
2. **Transfer gap is about 40×.** Standardized AP 0.004–0.006 versus real 0.2422.
3. **Multi-parent matching is broken; orphan cards are one symptom.** Orphans appear
   in 10 of the 15 training-seed-check runs, including the baseline at training
   seed 43 (11 orphans). Cause, in toolkit `handle_multi_parent`: the final `disp`
   keeps whichever parent-side version is smaller.
   - Client side smaller (baseline seed 42): unmatched accounts lose their
     dispositions (107/485 accounts have none).
   - Account side smaller: clients lose theirs, and cards (generated from the
     client-side version) point to missing dispositions (orphans) or to the wrong
     ones. FK checks cannot detect the wrong ones.
4. **Retraining moves results more than any setting.** Baseline across training
   seeds 42/43/44: ROC-AUC 0.614–0.769 (sd 0.081 vs ±0.007 across sampling seeds),
   prevalence 0.29–0.48%, accounts overdrawn 43.9–58.1%. Every 2026-10-29 value
   falls inside these ranges, so retraining noise, not the sampling-batch change,
   explains the difference between machines.

### Training-seed check — complete

Training seeds 42/43/44, sampling seed 42; each value retrains all models per seed.
Reports: [parent weight](../../results/experiments/transaction_overdraft_from_scratch/b52d0d6409bd1389/comparison.csv),
[clustering method](../../results/experiments/transaction_overdraft_from_scratch/4a98a4d7b7ff8dde/comparison.csv).
Cells show seed 42 / 43 / 44.

| Setting | ROC-AUC ↑ | Standardized AP ↑ | Before opening ↓ | Accounts ever overdrawn | Orphan cards |
| --- | --- | --- | --- | --- | --- |
| **Baseline** | 0.769 / 0.614 / 0.653 | 0.0056 / 0.0034 / 0.0036 | 20.5 / 23.1 / 16.1% | 58.1 / 51.8 / 43.9% | 0 / 11 / 0 |
| Parent scale 0.5 | 0.755 / 0.646 / 0.545 | 0.0050 / 0.0037 / 0.0027 | 22.5 / 21.7 / 24.6% | 65.2 / 56.7 / 31.3% | 0 / 16 / 12 |
| Parent scale 2.0 | 0.785 / 0.613 / 0.568 | 0.0054 / 0.0035 / 0.0025 | 22.7 / 19.8 / 15.3% | 65.2 / 57.5 / 39.8% | 0 / 10 / 38 |
| `gmm` | 0.774 / 0.630 / 0.704 | 0.0055 / 0.0034 / 0.0043 | 20.7 / 18.2 / 22.1% | 64.3 / 49.2 / 54.9% | 0 / 28 / 10 |
| `kmeans` | 0.730 / 0.621 / 0.552 | 0.0046 / 0.0037 / 0.0031 | 16.3 / 14.0 / 14.8% | 65.3 / 50.9 / 34.4% | 5 / 37 / 1 |

- **Parent scale 2.0:** its seed-42 gains reverse at seed 44 (ROC-AUC 0.568 vs
  0.653). Reject.
- **`gmm`:** ROC-AUC higher than the baseline in every seed (+0.005, +0.016, +0.051),
  which is small next to retraining sd 0.081.
- **`kmeans`:** fewer pre-opening transactions than the baseline in all 5 seed pairs
  tested, and better account→transaction correlation in every training seed. Lower
  ROC-AUC in 2 of 3 seeds. This is the only setting with a consistent chronology gain.
- The guidance sweep reused the training-seed-42 model, which scored best of the
  three (ROC-AUC 0.769 vs 0.614/0.653). Its conclusions hold for that model only.

### Multi-parent matching fix — implemented 2026-11-12

`tuning.match_multi_parent` replaces toolkit `handle_multi_parent` during sampling:

- It keeps the `disp` version that cards were generated from (client side).
- It gives every account one disposition first, then fills remaining account-side
  slots. No slot is used twice, so an account never gets more dispositions than it
  generated.
- Dispositions left without a slot are dropped. `_enforce_keys` then drops their
  clients and cards and counts every removal in each run's `matching_report.json`.

`completed_runs` now returns only runs from the current runner by default, so
pre-fix runs with identical configs are not mixed in.

Replayed on the 43 saved pre-matching tables (CPU, no regeneration):

| | Toolkit | Fixed |
| --- | ---: | ---: |
| Orphan cards | 193 | 0 |
| Clients without a disposition | 1,595 | 0 (dropped and counted) |
| Accounts without a disposition | 2,039 | 794 |
| Max dispositions per account (real 2) | 2 | 2 |
| Account→disp cardinality KS (mean) | 0.094 | 0.092 |

The remaining 794 accounts without a disposition all come from runs with fewer
synthetic clients than accounts. Each client has exactly one disposition, so not
every account can get one there. Client and account counts are generated
independently from district; fixing that is separate work. Dropping clients lowers
the mean client count from 559 to 522 (real 596) and cards from 81 to 75 (real 90).

### Matching-fix baseline on GPU — complete 2026-11-12

Baseline × training seeds 42/43/44, sampling seed 42. New runs `a406d8d664b2363e736c`,
`0e0e72cdfe1b2239e672`, `8a2b1c6eb4d7ee0ea6f3` (report `56bd7bd5670a77d7`), paired
with pre-fix `e2cf71…`, `fa627e…`, `524edf…` (report `b52d0d6409bd1389`).

| Training seed | 42 pre → fix | 43 pre → fix | 44 pre → fix |
| --- | --- | --- | --- |
| Clients (real 596) | 425 → 425 | 579 → 500 | 504 → 504 |
| Cards (real 90) | 59 → 59 | 74 → 61 | 57 → 57 |
| Orphan cards | 0 → 0 | 11 → 0 | 0 → 0 |
| Clients without disp | 0 → 0 | 79 → 0 (dropped) | 0 → 0 |
| Accounts without disp | 107 → 60 | 0 → 0 | 44 → 0 |
| Min FK valid rate | 1.0 → 1.0 | 0.851 → 1.0 | 1.0 → 1.0 |
| All-edge cardinality KS | 0.0867 → 0.0832 | 0.0680 → 0.0482 | 0.0561 → 0.0584 |
| Multi-parent corr RMSE | 0.0445 → 0.0480 | 0.0215 → 0.0193 | 0.0230 → 0.0128 |
| Multi-parent joint TVD | 0.1888 → 0.1868 | 0.1638 → 0.1652 | 0.1598 → 0.1581 |

- Structure is now valid in all 3 runs. Seed 42's remaining 60 accounts without
  a disposition are the client shortfall (425 clients for 485 accounts).
- Transaction, overdraft, chronology and transfer metrics are **identical** to
  pre-fix per seed (ROC-AUC 0.769/0.614/0.653, overdrawn 58.1/51.8/43.9%,
  pre-opening 20.5/23.1/16.1%). Matching does not touch `trans`, and the
  pre-matching tables are reproduced exactly.
- Decision: keep the fix. These 3 runs are the new baseline for step 3.

### Relative dates — implemented 2026-11-12, not yet run on GPU

- Top-level config `date_representation: absolute | relative_to_account`
  (default `absolute`; part of the clustering stage ID, so caches never mix).
  Launcher flag: `run_sweep --date-representation relative_to_account`.
- Encode (clustering stage only): `trans_date`, `loan_date`, `issued` become days
  since the account opened, after aligning epochs (shifts trans 0, loan 212,
  card 457 days). Cards reach their account through `disp`. Processed CSVs,
  evaluation, and the data fingerprint are unchanged.
- Real offsets are all ≥ 0: trans 0–2188, loan 108–693, card 211–2085.
- Decode after matching and key enforcement, using the synthetic parent's
  `account_date`. Negative offsets are clipped to 0 and counted in
  `matching_report.json → clipped_relative_dates`.
- Tests: exact real round trip, clipping, stage IDs differ. 49 pass; ruff clean.
- Editing `tuning.py` changed the runner hash, so the absolute baseline is rerun
  alongside (expected identical to the step-1 runs; doubles as a reproducibility check).

**GPU result (report `02dd0ec23c417328`, seeds 42/43/44, absolute → relative):**

- Absolute reruns are identical to the step-1 runs on every metric.
- Pre-opening transactions 20.5/23.1/16.1% → **0/0/0%**; no clipping needed.
- Days-since-opening KS 0.21/0.23/0.16 → 0.009/0.012/0.022. Loan vs opening
  date correlation (real 0.958) −0.115/0.038/−0.134 → 0.970/0.953/0.956.
- **New failure:** 24.5/23.6/19.4% of transactions fall after the data end
  (day 2188 = 1998-12-31). Calendar `trans_date` KS 0.005–0.044 → 0.19–0.25.
- Overdrawn accounts 58.1/51.8/43.9% → 31.3/51.9/27.0%; ROC-AUC
  0.769/0.614/0.653 → 0.540/0.636/0.549. Both are confounded by the
  out-of-window rows, so not interpreted.
- Decision: do not adopt as is. Next: encode dates as a fraction of the
  account's observable window (opening → data end), bounding both sides.

### Window-fraction dates (step 3b) — implemented 2026-11-12, not yet run on GPU

- `date_representation: fraction_of_window` stores
  `(date − opening) / (data end − opening)` in [0, 1]. Data end = last real
  child date = day 2188 (1998-12-31) in account-epoch days.
- Decode clips fractions to [0, 1] (counted in `clipped_relative_dates`).
  Synthetic accounts opened after the end get window 0, so their children land
  on the opening date.
- Real fractions all lie in [0, 1] and round-trip exactly. 51 tests pass; ruff clean.
- Runner hash changed again, so the absolute baseline is rerun alongside.

**GPU result (report `8351cbcb96323721`, seeds 42/43/44, absolute → window):**
runs absolute `77ea8a…`/`effdb1…`/`0cdcba…`, window `4ad3cb…`/`91773b…`/`df88b1…`.

- Absolute reruns identical to step 1 again.
- Before opening 20.5/23.1/16.1% → 0/0/0%; after data end 0 → 0; 0 clipped.
- Days-since-opening KS 0.21/0.23/0.16 → 0.13/0.12/0.09. Calendar `trans_date`
  KS 0.005/0.014/0.044 → 0.129/0.129/0.109; overall transaction KS ~0.056 → ~0.094.
- Loan vs opening correlation (real 0.958) → 0.90/0.84/0.82. Account→transaction
  correlation RMSE 0.110/0.134/0.093 → 0.029/0.027/0.019.
- Overdrawn accounts 58.1/51.8/43.9% → 47.1/47.7/40.0% (lower in all 3, still
  far from 5.6%). ROC-AUC 0.769/0.614/0.653 → 0.661/0.659/0.658; standardized AP
  unchanged (~0.003–0.006 vs real 0.2422).
- **Decision: adopt `fraction_of_window`.** Both chronology bounds hold by
  construction; the cost is a worse calendar-date marginal. These 3 runs are
  the new baseline for step 4 (overdraft concentration).

### Step 4 prep — 2026-11-12 (CPU only)

- `base_config()` now defaults to `fraction_of_window` (legacy stays `absolute`);
  `run_sweep` default matches. Runner hash changed: the baseline retrains next batch.
- Real cohort: all 500 accounts have transactions; 28 (5.6%) ever overdrawn. Even
  in those, only 3.6% of rows are negative. Minimum-balance 5th percentile is −439;
  median 600. Overdrawn share by `frequency`: 5.2% / 12.5% / 10.7%.
- Leak check: the detector uses only `trans_date`, `amount`, `trans_type`,
  `operation`, `k_symbol`, `bank`; the label is `balance < 0`. An account-level
  feature added inside the runner and dropped before capture never reaches
  evaluation, and the processed CSVs stay unchanged.

### Step 4 — `ever_overdrawn` account feature (implemented, awaiting GPU)

- Config `account_feature: none | ever_overdrawn` (default `none`); `run_sweep
  --account-features`. The flag is derived from real `balance < 0`, written to a
  copied `account_domain.json` under the clustering stage, and dropped from the
  saved synthetic accounts. `matching_report.json` records the synthetic flag rate
  (real: 0.056). `none` keeps earlier stage IDs; legacy mode rejects the flag.
- Test: `tests/test_account_feature.py` (real count 28; toolkit loads it).
- GPU result (seeds 42/43/44; baseline `41b419`/`de8f3f`/`20d6a7`, bit-identical
  to `4ad3cb`/`91773b`/`df88b1`; feature `72c160`/`f168cb`/`0c3b18`):
  - Synthetic flag rate 0.073 / 0.053 / 0.060 (real 0.056): the account model
    learns the flag.
  - Accounts with any negative balance: 47.1/47.6/40.0% → 46.4/48.2/31.0%
    (real 5.6%). Negative rows within those accounts stay ~0.5% (real 3.6%).
  - The flag reaches transactions only through cluster assignment, which is too
    weak to concentrate overdrafts. Decision: reject; keep `account_feature: none`.

### Running-balance post-processing — CPU check (rejected, 2026-11-12)

- Real balances are a running sum of signed amounts (`PRIJEM` +, others −):
  99.7% of account-days match up to within-day order. Recomputed from zero, real
  overdrawn accounts are 9.6% (vs 5.6%; within-day order).
- Recomputing synthetic balances (baselines `41b419`/`de8f3f`/`20d6a7`) makes it
  much worse: overdrawn accounts 47/48/40% → 85/75/83%, negative rows 0.3% →
  38/25/32%. Per-account credits and debits are not balanced, so sums drift.
- Decision: reject. Generated `balance` stays; no GPU run needed.

### Step 5 — clustering method (GPU done; CPU check, report pending)

Baseline `kmeans_and_gmm` `41b419`/`de8f3f`/`20d6a7`; `kmeans` `fc50b8`/`91edd6`/
`719235`; `gmm` `3c8eec`/`54428a`/`7a66c7` (seeds 42/43/44). Seed 42 was rerun
after a killed worker left two unfinished clustering folders (deleted).

| Seed 42 / 43 / 44 | baseline | kmeans | gmm |
| --- | --- | --- | --- |
| Accounts ever overdrawn (real 5.6%) | 47.1/47.6/40.0% | 50.6/48.1/28.2% | 50.5/47.5/38.2% |
| Calendar `trans_date` KS | .129/.129/.109 | **.045/.081/.081** | .088/.140/.110 |
| Days-since-opening KS | .126/.120/.086 | **.038/.071/.059** | .082/.130/.087 |
| Loan vs opening corr (real .958) | .897/.843/.820 | .768/.682/.777 | .857/.836/.799 |
| Transactions-per-account KS | .053/.045/.049 | .028/.055/.100 | .029/.058/.066 |

- `kmeans` improves both date KS in all 3 pairs but weakens the loan correlation in
  all 3; overdrafts are mixed. `gmm` is about equal to baseline.
- Before opening stays 0% everywhere.
- Report `a7b223c72a6c677d` (real-test ROC-AUC 0.872): detector ROC-AUC baseline
  .661/.659/.658, `kmeans` .672/.601/.553, `gmm` .678/.641/.639. Standardized AP
  is flat except `kmeans` seed 44 (.0100). Transaction KS: `kmeans` better 3/3
  (.062/.074/.078 vs .093/.090/.098).
- Decision: keep `kmeans_and_gmm`. `kmeans` hurts detector transfer in 2/3 seeds
  and the loan correlation in 3/3; `gmm` is within noise. Next: diffusion and
  classifier sweeps against the `fraction_of_window` baseline.

### Step 6 — diffusion iterations (GPU done; CPU check, report pending)

Baseline 5000 `41b419`/`de8f3f`/`20d6a7`; 2500 `37618f`/`6c2d49`/`294830`;
10000 `ddfb43`/`a9c894`/`2edb61` (seeds 42/43/44).

| Seed 42 / 43 / 44 | 2500 | 5000 (baseline) | 10000 |
| --- | --- | --- | --- |
| Accounts ever overdrawn (real 5.6%) | 96/62/90% | 47/48/40% | **41/31/33%** |
| Negative rows (real 0.21%) | 2.4/0.5/1.5% | 0.29/0.33/0.24% | **0.25/0.18/0.19%** |
| Calendar `trans_date` KS | .121/.110/.128 | .129/.129/.109 | .111/.130/.092 |
| Loan vs opening corr (real .958) | .80/.78/.73 | .90/.84/.82 | .89/.85/.89 |
| Transactions-per-account KS | .049/.046/.068 | .053/.045/.049 | .045/.027/.038 |

- Clear dose response: 10000 beats 5000 on overdrawn accounts in 3/3 pairs and is
  equal or better elsewhere; 2500 is much worse. Full report needed (ROC-AUC, AP).
- Report `edb433b7162d5fa9`: detector ROC-AUC 5000 .661/.659/.658 → 10000
  **.762/.680/.700** (3/3; seed 42 CI .737–.787 excludes baseline); standardized
  AP .0033/.0030/.0037 → .0040/.0033/.0039 (3/3). Transaction TVD .023/.023/.031 →
  .018/.014/.018; all-table numerical KS .097/.109/.131 → .076/.096/.093 (3/3).
  Account→transaction RMSE slightly worse in 2/3 (.027→.030, .019→.028).
  2500 is unstable (ROC-AUC .533/.726/.783; all-table KS ~.17).
- Decision: adopt 10000. `base_config()` now sets `diffusion_config.iterations =
  10000` for fresh training; `diffusion_steps_long` preset tests 20000. Runner hash
  changed, so the next batch retrains the baseline.

### Step 7 — diffusion iterations 20000 (done: rejected, keep 10000)

- Runs (seed 42/43/44): baseline 10000 901122/e3f6aa/c6e6c8 (overdrawn rate
  reproduces step 6 exactly); 20000 7da22f/65d9de/2eb9a1. All 6 runs completed at 15:46.
- CPU check (real: overdrawn accounts 5.6%, negative rows 0.21%):
  overdrawn accounts 41/31/33% → 25/28/24% (better in all 3 seeds); negative rows
  .25/.18/.19% → .12/.17/.12%; date KS .111/.130/.092 → .091/.125/.099 (2/3 better);
  balance KS mixed; transactions-per-account KS .045/.027/.038 → .050/.079/.020
  (worse in 2/3); loan amount KS .092/.094/.136 → .097/.135/.133 (worse in 2/3);
  transaction row count 97k/126k/140k → 100k/115k/100k (real 118k).
- Report 60063072d89dd766 (pairs 42/43/44, 10000 → 20000): ROC-AUC
  .762/.680/.700 → .691/.700/.680 (worse in 2/3, mean .714 → .690); standardized AP
  .0040/.0033/.0039 → .0040/.0045/.0040 (noise); transaction TVD
  .018/.014/.018 → .013/.014/.011; categorical TVD .096/.084/.100 → .091/.078/.076;
  numerical KS .076/.096/.093 → .085/.079/.078.
- Decision: keep 10000. Fidelity improves, but detector transfer does not, and
  training time doubles. `base_config()` is unchanged, so the next batch reuses
  the step 7 baseline stages (no retraining).

### Step 8 — diffusion presets (done: adopt batch size 512)

27 runs (9 settings × seeds 42/43/44) in parallel, 6 jobs per GPU on GPUs 1–5,
~50 min. Report `43e7f4d2b0d8d13d` also re-scores the step 7 runs; baseline values
match step 7 exactly. Cells: seed 42 / 43 / 44; "better" = pairs better than baseline.

| Setting | ROC-AUC | Std. AP | Txn TVD | Overdrawn accts | Verdict |
| --- | --- | --- | --- | --- | --- |
| baseline (lr 6e-4, 256 batch, 0 dropout, 500 t) | .762/.680/.700 | .0040/.0033/.0039 | .018/.014/.018 | 41/31/33% | – |
| **batch 512** | **.770/.691/.806 (3/3)** | **.0051/.0036/.0074 (3/3)** | **.012/.012/.013 (3/3)** | 38/29/42% (2/3) | **Adopt** |
| batch 128 | .797/.661/.686 (1/3) | .0077/.0030/.0040 | .026/.020/.023 | 65/39/39% | Reject |
| d_layers [256,512,512,256] | .731/.735/.709 (2/3) | .0037/.0035/.0034 | .021/.021/.020 | 62/66/57% | Reject |
| dropout 0.1 | .647/.648/.586 (0/3) | .0029/.0029/.0032 | .023/.024/.020 | 62/54/41% | Reject |
| dropout 0.2 | .724/.708/.695 (1/3) | .0038/.0032/.0028 | .038/.036/.033 | 91/76/93% | Reject |
| lr 0.0002 | .666/.547/.585 (0/3) | .0032/.0026/.0022 | .021/.020/.022 | 55/32/70% | Reject |
| lr 0.001 | .777/.671/.712 (2/3) | .0042/.0034/.0040 | .018/.015/.017 | 68/34/34% | Within noise |
| timesteps 250 | .708/.720/.693 (1/3) | .0041/.0039/.0042 | .019/.018/.016 | 32/34/36% | Within noise |
| timesteps 1000 | .669/.688/.740 (2/3) | .0038/.0078/.0045 | .017/.015/.018 | 35/36/43% | Within noise |

- Batch 512 also improves all-table numerical KS (2/3), categorical TVD (2/3) and
  cardinality KS (2/3). Mean ROC-AUC .714 → .755, std. AP .0037 → .0054.
- Dropout, lower LR and the smaller model clearly hurt; most spread overdrafts
  over more accounts.
- Decision: `base_config()` sets `diffusion_config.batch_size = 512`. Editing
  `tuning.py` changes the runner hash, so the next batch retrains its baseline.
  Next: classifier presets against this baseline.

### Step 9 — classifier presets (done: keep defaults)

Report `0d7c36bb65b1bacc`. Baseline (diffusion 10000 iterations, batch 512;
classifier lr 1e-4, 5000 iterations, d_layers [128..1024..128], batch 256, dim_t
128) reproduces step 8 batch 512 exactly. One change each, seeds 42/43/44;
"(n/3)" = seed pairs better than baseline.

| Setting | ROC-AUC | Std. AP | Transaction TVD | Overdrawn-rate gap |
| --- | --- | --- | --- | --- |
| baseline | .770/.691/.806 | .0051/.0036/.0074 | .0122/.0117/.0132 | .327/.237/.362 |
| lr 3e-5 | .770/.702/.832 (2/3) | .0045/.0055/.0077 (2/3) | (1/3) | .334/.252/.421 (0/3) |
| lr 3e-4 | .722/.675/.807 (1/3) | .0040/.0031/.0075 (1/3) | (1/3) | (0/3) |
| iterations 2500 | .754/.667/.825 (1/3) | .0042/.0033/.0069 (0/3) | (1/3) | (1/3) |
| iterations 10000 | .751/.664/.811 (1/3) | .0047/.0034/.0076 (1/3) | (0/3) | (0/3) |
| d_layers [128,256,256,128] | .798/.692/.825 (3/3) | .0052/.0037/.0073 (2/3) | .0138/.0139/.0132 (0/3) | (1/3) |
| batch 128 | .755/.697/.823 (2/3) | .0043/.0058/.0074 (1/3) | (0/3) | (1/3) |
| batch 512 | .776/.679/.825 (2/3) | .0048/.0040/.0076 (2/3) | (1/3) | (1/3) |
| dim_t 64 | .759/.694/.818 (2/3) | .0045/.0050/.0077 (2/3) | (1/3) | (1/3) |
| dim_t 256 | .792/.650/.827 (2/3) | .0049/.0032/.0078 (1/3) | (1/3) | (1/3) |

- No setting wins both transfer metrics in 3/3. Mean ROC-AUC range .735–.771
  (baseline .756); mean std. AP .0048–.0059 (baseline .0054): all within noise.
- Smaller classifier: ROC-AUC 3/3 (+.016 mean) but AP flat (+.0000) and
  transaction TVD worse 3/3. Not adopted.
- No setting improves overdraft concentration (best 1/3).
- Decision: keep classifier defaults; the classifier is not the bottleneck. No
  `tuning.py` change, so the runner hash and caches stay valid.

### Step 10 — final config, 5 training seeds (done)

Report `c480bd90c664b936`. Config: matching fix, `fraction_of_window`,
`kmeans_and_gmm`, diffusion 10000 iterations / batch 512, classifier defaults.
Seeds 42–44 reused from cache (identical to step 9); 45/46 new.

| Seed | 42 | 43 | 44 | 45 | 46 | Mean ± sd |
| --- | --- | --- | --- | --- | --- | --- |
| ROC-AUC (real .872) | .770 | .691 | .806 | .768 | .751 | **.757 ± .042** |
| Std. AP (real .242) | .0051 | .0036 | .0074 | .0055 | .0037 | **.0051 ± .0016** |
| Transaction TVD | .012 | .012 | .013 | .012 | .016 | .013 ± .002 |
| All-table numerical KS | .076 | .089 | .096 | .087 | .089 | .087 ± .008 |
| Accounts ever overdrawn (real 5.6%) | 38.3% | 29.3% | 41.8% | 38.9% | 29.4% | 35.5 ± 5.8% |
| Before opening / min FK validity | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 | 0 / 1.0 |

- New seeds land inside the 3-seed range: the step 8–9 numbers hold.
- ROC-AUC sd halved vs the first baseline (.081 → .042).
- Transfer gap: std. AP ~47× below real; overdraft spread unchanged.

### Overdraft diagnosis — CPU, final config seeds 42–46 (2026-11-13)

Script `/tmp/od_diag.py` (read-only on saved samples, report `c480bd90c664b936`).

| | Real | Synthetic (5 seeds) |
| --- | --- | --- |
| Negative rows overall | 0.21% | 0.17–0.28% (right amount) |
| Accounts with any negative row | 5.6% | 29–42% |
| Median negative rows per overdrawn account | 4 | 1 |
| Dispersion ratio (per-account neg-rate var / binomial) | 16.4 | 0.8–1.3 |
| Median negative balance | −2,356 | −2,500 to −23,000 |
| Median amount in negative rows | 235 | 15 |
| `k_symbol` 4 among negative rows (real overall 0.1%) | 25.5% | 0.0% |
| `operation` 4 among negative rows | 74.5% | 34.8% |

- Synthetic negative rows are scattered at random across accounts (dispersion ≈ 1,
  i.e. independent per row); real ones come in per-account episodes (16×).
- Their row profile is wrong: TVD vs real negative rows is 0.42 (`operation`) and
  0.38 (`k_symbol`). The rare `k_symbol` 4, which appears almost only on negative
  rows, is never generated. The detector learns these real signatures, so the
  synthetic labels sit on the wrong rows: this explains the low AP.
- Real median account balance does not predict overdraft (5–7% in every
  quintile), so the "account balance level" design (A) has a weak premise.

### Design D — rule-based overdraft episodes (CPU post-processing, 2026-11-13)

Code `experiments/analysis/overdraft_episodes.py`. Copies of the 5 final runs under
`results/experiments/overdraft_episodes/`; originals untouched. Scattered negative
balances made positive, then episodes placed in a real-like share of accounts
(count, length, balance size from real **train** accounts only). `episodes_profile`
also draws episode rows' `trans_type`/`operation`/`k_symbol` from the real joint
frequency of negative rows and amounts from a log-normal fit. Report
`overdraft_episodes/reports/32066784546dbc9d` (same real split and detector).

| Seeds 42–46 | original | episodes | episodes_profile |
| --- | --- | --- | --- |
| ROC-AUC (real .872) | .770/.691/.806/.768/.751 (.757) | .542/.515/.517/.511/.553 (.528) | .794/.767/.802/.789/.791 (**.789**) |
| Std. AP (real .242) | .0051/.0036/.0074/.0055/.0037 (.0051) | ~.0019 (random) | .193/.144/.204/.201/.190 (**.186**) |
| Overdrawn-account gap | .30 | **.001** | **.001** |
| Overdraft-row numerical KS | .40 | .20 | .19 |
| Overdraft-row categorical TVD | .24 | .23 | **.08** |
| Transaction KS / TVD, all-table KS / TVD | .086/.013, .087/.096 | unchanged | unchanged |

- `episodes` fixes concentration exactly but drops transfer to random: labels on
  random rows carry no signal. So the original weak AP came from the generator
  partly aligning labels with features.
- `episodes_profile` reaches AP .186 (77% of real). **This is circular**: the
  evaluation trains the detector on real data and scores synthetic data, and we
  stamped the real overdraft row profile onto the labelled rows, so the detector
  recognises its own signature. It does not show that a detector trained on this
  synthetic data works on real data.
- Decision: do not report D as a model improvement. Report it only as evidence
  that the gap is the overdraft row profile and episode structure. A fair test
  needs the reverse direction (train on synthetic, test on real), which the
  evaluator does not implement.

### Reverse test: train on synthetic, test on real (CPU, 2026-11-13)

Script `experiments/analysis/tstr_overdraft.py`; output `results/experiments/overdraft_episodes/tstr/tstr.csv`.
Same random forest and features as the evaluator, fit on each run's full synthetic `trans`,
scored on the same held-out real test accounts (split seed 42; 23,247 rows, prevalence
0.17%). Real→real reference: ROC-AUC .872, AP .242. One split and one forest seed, so
the spread below covers generation seeds only.

| Seeds 42–46 | ROC-AUC | AP (random ≈ .0017) |
| --- | --- | --- |
| original | .895/.872/.899/.906/.865 (**.887 ± .018**) | .060/.107/.076/.108/.072 (**.085 ± .022**) |
| episodes | .719/.598/.669/.636/.746 (.674 ± .060) | .0034/.0021/.0026/.0025/.0049 (.003) |
| episodes_profile | .850/.780/.803/.802/.826 (.812 ± .027) | .250/.204/.254/.210/.198 (**.223 ± .027**) |

- **The unmodified generator is far more useful than the real→synthetic metric shows.**
  A detector trained only on synthetic data reaches real→real ROC-AUC (.887 vs .872)
  and 35% of real AP (50× random). Real→synthetic AP (.005) mostly penalises how
  synthetic labels are spread, not whether the synthetic data teaches the right rule.
- `episodes` hurts in 5/5 seeds: moving labels to random rows destroys the signal.
- `episodes_profile` raises AP in 5/5 seeds (92% of real) but lowers ROC-AUC in 5/5. It is
  not leakage (rules calibrated on real train accounts only; test accounts unseen), but
  the gain comes from hand-written rules, so report it as "generator + rules", not as the
  generator.

### Decided next steps

1. Pause hyperparameter sweeps; defer the diffusion and classifier presets.
2. Fix multi-parent matching: every synthetic account and client keeps a
   disposition, and cards stay attached to the disposition they were generated
   from. Count and report anything dropped.
3. Store transaction, loan, and card dates as days since account opening; convert
   back to calendar dates after generation so evaluation is unchanged.
4. Then address overdraft concentration. Candidate (untested): an account-level
   feature such as minimum balance, so clustering groups overdraft-prone accounts.
5. New baseline with fixes 2–3 and 3 training seeds; compare with the current
   baseline paired by training seed. Any setting that retrains needs 3 training
   seeds from now on.

## Implementation progress

- [x] Reuse the supplied overdraft evaluator and toolkit training/sampling.
- [x] Implement shared transaction/account quality and bootstrap diagnostics.
- [x] Implement per-stage configs, fingerprints, status, and artifact reuse.
- [x] Supply 15 parameter notebooks, a generic template, and final comparison.
- [x] Add `01_baseline.ipynb` with input checks and no saved-model dependency.
- [x] Remove historical-run discovery and static outcome claims from current notebooks.
- [x] Use dedicated `transaction_tuning_from_scratch` and
  `transaction_overdraft_from_scratch` roots across the workflow.
- [x] Include the matched baseline parameter value for each requested seed pair;
  deduplicate candidates before spending GPU work.
- [x] Guard generation with `RUN_NEW_TRIALS = False` and require a matching completed
  baseline before a parameter sweep is enabled.
- [x] Handle zero/one completed runs in evaluation without assuming results exist.
- [x] User: run `00_prepare_account_sample.ipynb` to create the selected processed cohort.
- [x] User: execute the baseline; record actual configs, runtime, and results.
- [x] User: complete and evaluate the first guidance sweep; record its tradeoffs.
- [x] Sampling batch 8192 and concurrency-safe stage locks in `tuning.py`.
- [x] Add `run_sweep.py` for multi-GPU sweeps.
- [x] User: rerun cohort, baseline, and four sweeps with 3 sampling seeds on the H200 node.
- [x] User: training-seed check (seeds 42/43/44) for baseline, parent scale, clustering method.
- [x] Fix multi-parent matching (missing dispositions, orphan/misattached cards); replay-validated.
- [ ] User: baseline × training seeds 42/43/44 with the matching fix only.
- [ ] Store dates relative to account opening; new baseline.
- [ ] Investigate account-level overdraft concentration and pre-opening transactions.
- [ ] User: execute later selected sweeps/repeats after the investigation.

## Research decisions

1. **Target:** current `balance < 0`, using the supplied random forest and
   account-grouped detector split. Balance, IDs, and target-derived features are
   excluded from predictors. This is current-event detection.
2. **Transfer:** real-trained detector → synthetic data, alongside held-out real
   accounts. An independent synthetic-trained → real-test utility study needs
   outer account exclusion before generator clustering/training; internal model
   splits alone do not provide that separation.
3. **Rare events:** always report counts, prevalence, and uncertainty. AP alone
   is affected by event prevalence and task difficulty. Prior-standardized AP is
   supplementary; no automatic winner or combined weighted score is used.
4. **Both quality views:** all 8 tables and direct relationships, plus transaction/
   account and overdraft-conditional checks. Exclude identifiers from numerical
   fidelity metrics. Loan table distributions are assessed; loan classification
   is not a downstream task.
5. **Fresh baseline:** align diffusion/classifier internal splits to 70/20/10
   because they fit preprocessing independently. Keep this baseline and full sample
   scale fixed when comparing a parameter. Prior model checkpoints are not used.

## Notebook and implementation decisions

- The user explicitly requested one notebook per parameter and will execute them.
  Helpers contain shared logic; notebooks contain settings and presentation.
- `01_baseline.ipynb` trains all required stages; running the original training or
  synthesis notebooks separately is not required.
- Original notebooks/configs/artifacts remain references. A separate new root
  makes the starting state independent of older experiment files.
- Parameter sweeps include their baseline even when `VALUES` omits it. Each seed
  pair gets its own matched control. Identical completed stage hashes are reused.
- The final comparison uses one fitted detector and real account split across all
  selected completed runs. Zero runs produce instructions; one run is a reference
  report, not evidence of a hyperparameter winner.
- The current evaluation notebook is general: discover new-workflow runs rather
  than load five fixed paths. Static numerical conclusions have been replaced by
  blank prompts to fill after execution. The fresh baseline and guidance notebooks
  now contain conclusions grounded in their completed saved reports.
- No additional tracking service, database, or automatic optimizer is introduced.

## Remaining research limitations

- The user completed fresh ClavaDDPM training/sampling and evaluation. Recorded
  baseline stages total 34 min 08 s, excluding evaluation; generation alone took
  24 min 10 s. Different settings/hardware may have different costs.
- Toolkit classifier accuracy uses a fixed batch-size denominator, including short
  batches; do not use that logged accuracy to select models.
- Transaction/account dates have different epochs; quality checks align them.
- The new cohort retains complete observed raw histories for selected accounts.
  Generated rows are not guaranteed to form faithful sequences. Leakage-safe
  rolling features, balance recurrence, and graph metrics remain separate work.
- Bootstrap intervals condition on a fitted detector and saved generation. They
  do not replace generation seeds or model-training repeats.
- Some toolkit splits/random states are fixed; a training seed does not change
  every internal partition. Cross-hardware determinism is not guaranteed.
- Failed stages record their state but do not support optimizer-state resume.

## Completed baseline — 2026-10-29 (previous machine)

Superseded by the H200 rerun above; these artifacts are not on the current machine.

The user successfully executed `01_baseline.ipynb` and created the 500-account
cohort. All baseline stage statuses and the evaluation report are `complete`.
Only the baseline notebook's Markdown and this tracker were updated when recording
these results; its executed code/settings/output cells and other notebooks were
preserved.

- **Dataset:** `data/berka_accounts_500_seed42`, 500 accounts and 117,560 transactions;
  247 overdraft rows across 28 accounts, all 8 linked tables.
- **Baseline sample:** `d32c2dce062b79cbbdc4`; model set `1f6590126148e13f5b33`.
- **Evaluation report:** `results/experiments/transaction_overdraft_from_scratch/64f66493e1c5e4a1`.
- **Controls:** guidance 1, training/sampling seeds 42/42, 50 K-means + GMM clusters,
  parent scale 1, full sample scale 1, 5,000 updates for diffusion/classifiers,
  500 timesteps, and aligned 70/20/10 internal splits. See the saved `config.yaml`
  for full architectures and settings.

### Recorded stage times

| Stage | Seconds | Approximate duration |
| --- | ---: | ---: |
| Clustering | 20.0 | 20 s |
| Diffusion models | 226.9 | 3 min 47 s |
| Guidance-classifier/model assembly | 352.0 | 5 min 52 s |
| Synthesis, matching, and saving | 1,449.6 | 24 min 10 s |
| Recorded total | 2,048.5 | 34 min 08 s |

Device is recorded as CUDA. These timers exclude detector evaluation and work
outside the stage callbacks; they are not total notebook wall-clock duration.

### Overdraft transfer

The fixed real detector trains on 400 accounts / 94,313 rows / 208 overdrafts.
Its real test has 100 accounts / 23,247 rows / 39 overdrafts, concentrated in only
**5 accounts**. The synthetic baseline has 469 accounts / 114,915 rows / 351 overdrafts.

| Metric | Real test | Synthetic baseline |
| --- | ---: | ---: |
| Event prevalence | 0.1678% | 0.3054% |
| AP | 0.2422 | 0.00595 |
| Raw AP 95% account-bootstrap interval | 0.0005–0.3606 | 0.0052–0.0071 |
| ROC-AUC | 0.8722 | 0.6249 |
| AP standardized to real-test prevalence | 0.2422 | 0.00327 |

Full-cohort event prevalence is 0.2101%. Bootstrap settings are seed 42 and 200
draws; 199 real and 200 synthetic draws contain both classes. The wide real AP
interval reflects limited independent positive accounts. The point estimates and
fidelity failures warrant further study; do not claim a precise or statistically
established transfer-gap magnitude from this single fit/seed.

### Fidelity and integrity findings

- All 8 tables and 8 edges were scored. Minimum PK uniqueness and FK validity are
  1.0; no invalid measured values, unknown categories, or undefined marginal scores
  were detected. These checks do not establish useful synthetic behavior.
- Table-averaged numerical KS / categorical TVD are **0.0914 / 0.1043**.
  Transaction KS/TVD are **0.0589 / 0.0229**, but overdraft-only KS/TVD are
  **0.3829 / 0.3958**.
- Overdraft concentration is substantially different: **215/469 synthetic accounts
  (45.8%)** versus **28/500 real accounts (5.6%)** ever overdraw. Transaction-level
  prevalence alone misses this loss of account-level concentration.
- **19.0%** of synthetic transactions occur before account opening, versus **0%**
  of real transactions. This remains a central target for subsequent comparisons.
- Notable attribute errors include district categorical TVD **0.3052** and
  account→loan correlation RMSE **0.2978**. Multi-parent client/account mixed
  correlation RMSE is **0.0649**, with joint categorical TVD **0.1641**.

| Table | Real rows | Synthetic rows |
| --- | ---: | ---: |
| district | 77 | 77 |
| client | 596 | 435 |
| account | 500 | 469 |
| disp | 596 | 435 |
| card | 90 | 59 |
| loan | 76 | 52 |
| order | 742 | 807 |
| trans | 117,560 | 114,915 |

Client/disposition counts are 27.0% below reference, cards 34.4% below, and loans
31.6% below. These are measured discrepancies, not an attribution to matching or
another specific component without a separate causal check.

**Undefined global relationship RMSE:** the two `disp→card` attribute pairs cannot
be scored because disposition type is constant among card-linked rows in both
real and synthetic data. Their missing values propagate to the aggregate by
design. They are not zero error or evidence of an execution failure. Use the
individual valid edge scores; do not silently turn this `NaN` into a numeric score.

### Baseline follow-up

The guidance comparison below has now completed using the same cohort, seeds,
and model set. The earlier running status for guidance 0 is superseded by its
verified complete status. The baseline remains unchanged; there is no confirmed
hyperparameter winner or independent synthetic-trained→real utility result.

## Completed guidance sweep — 2026-10-29 (previous machine)

Superseded by the H200 rerun above; report links below point to files not on this machine.

The user completed all five settings and the shared CPU evaluation in
`guidance_strength_tuning.ipynb`. All five sample statuses and report status are
`complete`. Saved manifests have identical data/backend contexts and configs
apart from `sampling_config.classifier_scale`; training/sampling seeds are 42/42.
Clustering, diffusion, and classifier models were reused from the fresh baseline.

**Report:** [comparison.csv](../../results/experiments/transaction_overdraft_from_scratch/e137bf61d4d4fdb2/comparison.csv).
One fitted real detector and account split were used across all five settings.

| Guidance | Saved sample | Generation/matching time |
| --- | --- | ---: |
| 0 | `5e1ee60aff45370f4680` | 24.73 min |
| 0.5 | `9d1bd199c0b8ffe393c4` | 25.76 min |
| 1 (baseline) | `d32c2dce062b79cbbdc4` | 24.16 min |
| 2 | `a3860d6c1335b858616c` | 25.74 min |
| 4 | `e7a4c18ed1c24b395ec4` | 25.41 min |

The four new generations total **101 min 39 s** of recorded sampling/matching/
saving time; baseline sampling was cached. These stage timers exclude evaluation
and other notebook overhead, not total wall-clock runtime.

### Transfer and fidelity evidence

| Guidance | Raw AP ↑ | Prevalence-adjusted AP ↑ | ROC-AUC ↑ | Overdraft rows | Overdraft prevalence |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.00684 | 0.00390 | 0.6664 | 331 | 0.2952% |
| 0.5 | 0.00682 | 0.00407 | 0.6678 | 330 | 0.2816% |
| 1 (baseline) | 0.00595 | 0.00327 | 0.6249 | 351 | 0.3054% |
| 2 | 0.00675 | 0.00307 | 0.6156 | 431 | 0.3694% |
| 4 | 0.01220 | 0.00333 | 0.6244 | 716 | 0.6246% |

Adjusted AP uses the real-test prevalence. Real-test AP is 0.2422 and ROC-AUC
0.8722; the full real cohort has 0.2101% overdraft transactions.

| Guidance | Transaction numerical KS ↓ | All-table numerical KS ↓ | All-table categorical TVD ↓ | Accounts ever overdrawn | Transactions before account opening |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0502 | 0.0949 | 0.0879 | 49.5% | 20.7% |
| 0.5 | 0.0562 | 0.0887 | 0.0938 | 46.5% | 19.8% |
| 1 (baseline) | 0.0589 | 0.0914 | 0.1043 | 45.8% | 19.0% |
| 2 | 0.0758 | 0.0977 | 0.1076 | 47.3% | 18.6% |
| 4 | 0.1227 | 0.1244 | 0.1226 | 48.2% | 22.7% |

KS/TVD are lower-is-better. Real account-overdraft rate is 5.6% and pre-opening
transaction rate is 0%. All five runs score all eight tables/edges with minimum
PK uniqueness and FK validity 1.0. Global relationship RMSE remains missing due
to two undefined `disp→card` pairs; individual defined edge metrics remain usable.

### Decision and limitations

- **Retain 0/0.5 as candidates, not winners.** Setting 0 has the lowest transaction
  numerical KS, all-table categorical TVD, and cardinality KS. Setting 0.5 has the
  highest adjusted AP/ROC-AUC point estimates and lowest all-table numerical KS.
  Its adjusted AP is about 24% above baseline, but overlapping bootstrap intervals
  and one seed do not establish a repeatable gain.
- **Do not select 4 based on raw AP.** It raises overdraft prevalence to 0.6246%;
  adjusted AP is nearly the baseline value and ROC-AUC is unchanged in practice.
  Its overdraft-only numerical KS improves to 0.2927 from baseline 0.3829, while
  overall distributions and chronology worsen. Setting 2 reduces pre-opening
  transactions to 18.6%, but that still conflicts with the real reference.
- **Keep baseline 1 frozen; investigate before broader tuning.** Every setting
  disperses overdrafts across too many accounts (45.8–49.5% versus real 5.6%) and
  generates pre-opening transactions (18.6–22.7% versus real 0%). Valid keys and
  similar marginal distributions are insufficient evidence of realistic histories.

The real test has only five positive accounts and a wide AP interval
(0.0005–0.3606). The 200 account-bootstrap draws condition on one detector fit
and saved generation; they do not capture training/generation-seed variation.
Do not claim statistical superiority, a precise transfer-gap magnitude, or a
confirmed improvement in downstream real-data utility.

### Next investigation — planned, not yet performed

1. Verify date decoding/alignment and trace how account attributes condition
   transaction generation; distinguish representation errors from generated values.
2. Compare existing before/after-matching artifacts to locate when timing and
   overdraft concentration diverge. Do not attribute the issue to matching without
   stage-level evidence, and do not delete or regenerate saved artifacts.
3. Use the findings to choose the next targeted experiment. Defer broad clustering/
   training sweeps; reconsider 0/0.5 repeats with matched baseline controls after
   the audit. The user remains responsible for any notebook execution.

Only the guidance notebook's final Markdown conclusion and this tracker were
edited for this documentation update. Executed code, settings, outputs, other
notebooks, datasets, and saved model/synthetic/report artifacts were preserved.

## Development validation history (not inputs to this workflow)

Earlier development evaluated five older guidance artifacts and passed 28 workflow
regression checks in a restored environment. Those files and their report at
`../results/experiments/transaction_overdraft/0cd49913cd93fca6/` are historical;
they are not discovered, used as a baseline, or assumed by this run order.
The original `guidance_strength_original_checkpoints.ipynb` retains the older experiment record.

The user executes the current notebooks. Latest checks for this adaptation:

- **44 regression tests passed** with GPU visibility disabled. Coverage includes
  empty discovery, baseline inclusion for scalar/network candidates and seed pairs,
  account separation, target exclusion, complete account-cohort selection, non-destructive
  preparation/reuse, all-table metrics, shared-report exports, dataset mismatch
  rejection, and mocked stage reuse.
- **20 workflow notebooks** pass notebook-format/Python-syntax checks. Generation
  and preparation writes are disabled by default, parameter sweeps require a
  completed matching baseline, and evaluation guards zero-run/missing-cohort state.
  No cells were executed by the agent during those checks; the user subsequently
  completed the baseline and guidance sweep.
- Ruff, `git diff --check`, and local notebook/README link checks pass.
- The earlier orchestration tests used substitutes. The user has now also
  completed a real baseline training/synthesis/evaluation run and the first
  guidance sampling sweep. Broader parameter sweeps remain deferred.


## Account-cohort and full-table implementation decisions

- `dataset_settings.py` is the shared, editable location for **500 accounts, seed
  42**, raw input path, and processed `DATA_DIR`. The preparation notebook exposes
  overrides; every experiment shows its selected input. Restart kernels after
  editing shared defaults. A changed cohort needs its own matching baseline.
- `account_sampling.py` follows account-linked transactions/loans/orders/
  dispositions, then referenced cards/clients/districts. It reuses the existing
  preprocessor via an optional `raw_tables` input; its original file-based behavior
  is retained. Other accounts of a shared client are not pulled into the sample.
- Preparation has no transaction-row cap and does not oversample overdraft cases.
  It writes selected IDs, cohort counts, source/config/code fingerprints, and
  artifact checksums. Complete identical outputs are verified/reused; untracked,
  changed, failed, or incompatible output directories are not overwritten.
- `multi_table_quality.py` reuses KS/TVD and toolkit mixed associations for all
  declared tables/edges. It checks key validity, child counts including zeros,
  join coverage, within/cross-table attributes, and multi-parent `disp` matching.
  `trans.account` and `order.account_to` are identifier exclusions.
- Multi-parent checks include joint categorical TVD because association strengths
  can miss specific parent-category combination changes. Per-pair and per-table/
  edge diagnostics are saved; no task/fidelity aggregate winner is used.
- Full-reference CSVs, domains, schema, and available cohort metadata are now
  fingerprinted. Trial/reference input mismatches are rejected before scoring.
  Discovery filters completed runs by the selected dataset.

## Read-only default-source preview

The source files currently available produce the following cohort for 500 accounts
and seed 42; this was checked in memory without writing the prepared dataset:

| Table | Selected rows |
| --- | ---: |
| account | 500 |
| client | 596 |
| disp | 596 |
| district | 77 |
| card | 90 |
| loan | 76 |
| order | 742 |
| trans | 117,560 |

There are **247 overdraft transaction rows across 28 accounts**. These are
preparation diagnostics, not model/evaluation outcomes. Events are concentrated
within accounts, so event-row count alone overstates independent evidence.
Small card/loan tables can make 50-cluster guidance fragile; inspect the baseline
and cluster-count experiments rather than silently changing the baseline.

The default processed cohort directory was subsequently created by the user,
who also completed the baseline notebook. The agent did not execute those notebooks.
Existing datasets, model checkpoints, synthesized results, and the historical
notebook remain untouched by this extension.


## Handoff status

Implementation and CPU/static validation for the approved extension are complete.
All 44 tests pass with GPU visibility disabled; Ruff, notebook format/syntax,
local documentation links, and diff whitespace checks pass. The shared report was
exercised with small test fixtures and exports both metric views and plots.

**Current next step (2026-11-12):** run the baseline with the matching fix only
(training seeds 42/43/44), then store dates relative to account opening and build
the next baseline. Keep the current baseline frozen. Diffusion and classifier
sweeps are deferred. Notebook execution remains the user's responsibility.
