# Transaction and overdraft tuning — progress and decisions

Updated: 2026-10-29. **Current workflow starts with a new account cohort and no experiments.**
The user runs notebooks. No training, generation, or notebook execution was
launched for this adaptation.

## Approved extension — implemented; user execution pending

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

Build the first baseline, then compare one hyperparameter at a time using the
supplied overdraft-transfer evaluator, transaction/account diagnostics, and full-table quality.
No existing generator checkpoints or historical synthetic runs are prerequisites.

1. Restore the tabular environment; run `prepare_account_sample.ipynb` on CPU
   to preview and prepare 500 accounts, seed 42, in a new output directory.
2. Run `00_baseline.ipynb`: preview, explicitly enable GPU work, train/generate,
   and evaluate one full baseline (training seed 42, sampling seed 42).
3. Run `guidance_strength_tuning.ipynb` using the new baseline models.
4. Explore individual clustering, diffusion, and guidance-classifier notebooks;
   repeat promising settings across seeds after a first-pass comparison.
5. Run `all_experiments.ipynb` on CPU for the combined comparison; fill conclusions.

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
- [ ] User: run `prepare_account_sample.ipynb` to create the selected processed cohort.
- [ ] User: execute the baseline; record actual configs, runtime, and results.
- [ ] User: execute selected parameter sweeps; record tradeoffs and next decisions.

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
  blank prompts to fill after execution.
- No additional tracking service, database, or automatic optimizer is introduced.

## Limitations for the first run

- Full fresh ClavaDDPM training/sampling has not been executed by this workflow.
  Inspect baseline status/output before launching a larger sweep. Training time
  has not yet been measured; do not assume a budget from generation alone.
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

## Results — fill after your baseline

- Baseline run/config: pending.
- Training and generation runtime: pending.
- Real train/test event counts and uncertainty: pending.
- Baseline transfer and fidelity diagnostics: pending.
- First guidance comparison and next decision: pending.

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
  completed matching baseline, and evaluation guards zero-run/missing-cohort state. No notebook cells were executed.
- Ruff, `git diff --check`, and local notebook/README link checks pass.
- Real fresh generator training/sampling remains for the user's baseline run;
  orchestration tests use substitutes for those expensive stages.


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

The default processed cohort directory has not been created by the agent. The
user runs `prepare_account_sample.ipynb`, then baseline/parameter notebooks.
Existing datasets, model checkpoints, synthesized results, and the historical
notebook remain untouched by this extension.


## Handoff status

Implementation and CPU/static validation for the approved extension are complete.
All 44 tests pass with GPU visibility disabled; Ruff, notebook format/syntax,
local documentation links, and diff whitespace checks pass. The shared report was
exercised with small test fixtures and exports both metric views and plots.

**Next user action:** open `prepare_account_sample.ipynb`. Review the default
500-account paths/settings, preview on CPU, then enable `PREPARE_DATASET` to write.
Run `00_baseline.ipynb` afterward. No preparation notebook, training notebook,
synthetic generation, or new GPU experiment was executed by the agent.
