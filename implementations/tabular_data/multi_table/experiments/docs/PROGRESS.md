# Transaction and overdraft tuning — progress and decisions

Updated: 2026-10-29. **Account cohort preparation, the fresh baseline, and all five guidance settings with shared evaluation are complete.**
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

The baseline and guidance comparison are complete. Investigate the observed
account-behaviour and chronology discrepancies before expanding tuning. Keep the
supplied overdraft-transfer evaluator, transaction/account diagnostics, and full-table quality.
No existing generator checkpoints or historical synthetic runs are prerequisites.

1. **Completed:** environment and `prepare_account_sample.ipynb`; 500-account
   cohort, seed 42, written to its own dataset directory.
2. **Completed:** `00_baseline.ipynb`; training, generation/matching, and evaluation
   for the fresh guidance-1 baseline, training/sampling seeds 42/42.
3. **Completed:** `guidance_strength_tuning.ipynb`; values 0, 0.5, 1, 2, 4,
   seeds 42/42, shared baseline models and one evaluation detector.
4. **Next:** investigate account conditioning/assignment and chronology using saved
   artifacts; these are candidate explanations, not confirmed causes.
5. **Deferred:** broader parameter sweeps, seed repeats, and the final combined
   comparison until the audit identifies a useful next experiment.

See `README.md` for the notebook index and detailed handoff.

## Implementation progress

- [x] Reuse the supplied overdraft evaluator and toolkit training/sampling.
- [x] Implement shared transaction/account quality and bootstrap diagnostics.
- [x] Implement per-stage configs, fingerprints, status, and artifact reuse.
- [x] Supply 15 parameter notebooks, a generic template, and final comparison.
- [x] Add `00_baseline.ipynb` with input checks and no saved-model dependency.
- [x] Remove historical-run discovery and static outcome claims from current notebooks.
- [x] Use dedicated `transaction_tuning_from_scratch` and
  `transaction_overdraft_from_scratch` roots across the workflow.
- [x] Include the matched baseline parameter value for each requested seed pair;
  deduplicate candidates before spending GPU work.
- [x] Guard generation with `RUN_NEW_TRIALS = False` and require a matching completed
  baseline before a parameter sweep is enabled.
- [x] Handle zero/one completed runs in evaluation without assuming results exist.
- [x] User: run `prepare_account_sample.ipynb` to create the selected processed cohort.
- [x] User: execute the baseline; record actual configs, runtime, and results.
- [x] User: complete and evaluate the first guidance sweep; record its tradeoffs.
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
- `00_baseline.ipynb` trains all required stages; running the original training or
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

## Completed baseline — 2026-10-29

The user successfully executed `00_baseline.ipynb` and created the 500-account
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

## Completed guidance sweep — 2026-10-29

The user completed all five settings and the shared CPU evaluation in
`guidance_strength_tuning.ipynb`. All five sample statuses and report status are
`complete`. Saved manifests have identical data/backend contexts and configs
apart from `sampling_config.classifier_scale`; training/sampling seeds are 42/42.
Clustering, diffusion, and classifier models were reused from the fresh baseline.

**Report:** [comparison.csv](../results/experiments/transaction_overdraft_from_scratch/e137bf61d4d4fdb2/comparison.csv).
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
The original `guidance_strength.ipynb` retains the older experiment record.

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

**Current next step:** investigate account-level overdraft concentration and
pre-opening transactions using saved artifacts. Preparation, baseline, and the
five-setting guidance comparison are complete; conclusions are recorded. Keep
guidance 1 frozen and 0/0.5 as candidates, with no confirmed winner. Defer broader
sweeps until the audit informs the next experiment. No reruns are needed for this
documentation update; notebook execution remains the user's responsibility.
