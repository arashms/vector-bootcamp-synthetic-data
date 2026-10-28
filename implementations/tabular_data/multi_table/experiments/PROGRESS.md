# Transaction and overdraft tuning — progress and decisions

Updated: 2026-10-28. This is the working research record, not a claim of a finished experiment.

## Objective and agreed scope

Build a notebook-led workflow for guidance, clustering, diffusion, and classifier
hyperparameter experiments. Judge the generated transaction table using the
existing overdraft-transfer evaluator plus transaction and account-level quality
checks. Evaluate the five saved guidance runs first. **No new GPU sweep is
authorized for this stage.**

## Progress

- [x] Inspect the existing evaluator, saved trials, preprocessing, and toolkit APIs.
- [x] Restore the dependencies already declared in the tabular-data group.
- [x] Implement reusable transaction-quality and uncertainty diagnostics.
- [x] Implement reusable experiment planning, checkpoint reuse, and execution.
- [ ] Add notebooks for saved-run evaluation and future tuning.
- [ ] Evaluate all five saved runs, validate the workflow, and record findings.

## Research decisions

1. **Task:** detect negative balance on a transaction (`balance < 0`). Reuse the
   supplied random forest and account-grouped split. Balance, IDs, and features
   derived from the target remain outside classifier inputs. This is detection,
   not a prediction of a future overdraft.
2. **Evaluation direction:** start with real-trained detector → synthetic data.
   Treat this as transfer and fidelity diagnostics. A later synthetic-trained
   detector → real test requires a real test set excluded before clustering and
   generator training. Existing checkpoints do not establish that separation.
3. **Rare events:** the current reference has 51 overdrafts / 20,000 transactions
   (0.255%). Saved synthetic runs have approximately 0.41–0.57%. Report prevalence,
   event counts, and account-level bootstrap intervals with classification scores.
   Do not choose a winner solely by the highest synthetic average precision.
4. **Scope of quality:** focus on `trans` and `account → trans`, including
   distributions conditional on overdraft. Counterparty/account numbers are IDs,
   not continuous financial measurements. Loan metrics will not rank these runs.
5. **Training consistency:** the original diffusion split is 99/0.5/0.5 while the
   classifier defaults to 70/20/10. Both independently fit preprocessing on their
   own training rows. Fresh training experiments will use the same 70/20/10 split
   for both models, with a separate fresh baseline. Existing checkpoints stay a
   legacy baseline; changing that split is not attributed to a guidance effect.

## Implementation decisions

- Keep original training/synthesis notebooks and baseline artifacts as references.
- Reuse toolkit training/sampling functions and the supplied overdraft evaluator.
- Add small importable helpers for code shared by notebooks. Do not add a service,
  database, experiment-tracking dependency, or automatic optimizer.
- Identify reusable artifacts by their data, resolved config, seed, and source
  fingerprints. Record partial failures; never silently reuse incompatible work.
- Keep model training separate from sampling so guidance trials share models.
- Save readable plans, resolved configs, metrics, plots, and per-run status.

## Known limitations to address or report

- Toolkit classifier accuracy divides by a fixed batch size, including short
  batches. Do not use that logged accuracy to select models.
- Transaction dates and account dates have different epoch offsets. Align epochs
  before checking chronology.
- Transactions are a random subsample, so account histories are incomplete.
  Defer rolling/history predictors and sequential balance identities.
- One generation seed cannot establish a reliable winner. Bootstrap intervals
  estimate uncertainty conditional on a fitted detector; they do not replace
  generation seeds or prove generalization.

## Findings and validation

First CPU evaluation completed for all five saved guidance runs. Validation and
notebook integration are in progress.

- The real detector holdout contains **8 overdrafts / 3,917 transactions**; train
  contains 43 / 16,083. The real test AP estimate is 0.170 with a very wide
  account-bootstrap interval (approximately 0.002–0.557, 200 replicates).
- Synthetic AP ranges from 0.014 to 0.038. Guidance zero has the highest raw AP;
  that alone does not establish the best synthetic dataset.
- Guidance 0.5 has the lowest overall transaction numerical KS and categorical
  TVD. Guidance 4 worsens these marginals substantially.
- **20.6–24.8% of generated transactions predate their account opening** after
  aligning the distinct date epochs; real data has zero. Valid foreign keys do
  not guarantee realistic relationships.
- Overdraft-conditional numerical KS is roughly 0.43–0.53 despite overall
  transaction KS of 0.05–0.09: reasonable marginals conceal rare-event failures.

Next: finish notebooks, run regression tests, and record final report location.
