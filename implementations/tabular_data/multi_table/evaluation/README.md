# Multi-table quality and overdraft transfer evaluation

[`overdraft_transfer_evaluator.py`](overdraft_transfer_evaluator.py) trains a random
forest on real transactions, then evaluates the same detector on synthetic
transactions. The real train/test split keeps accounts separate (80%/20% by
default). The target is `balance < 0`; balance and IDs are excluded from predictors.

Inputs can be CSV paths or pandas DataFrames. Both need `trans_date`, `amount`,
`trans_type`, `operation`, `k_symbol`, `bank`, and `balance`; real data also needs
`account_id`. Use the same preprocessed date format and categorical code mappings
for real and synthetic data.

## Usage

Run from this evaluation directory so the local import works. Your Python
environment needs NumPy, pandas, and scikit-learn.

```python
from overdraft_transfer_evaluator import OverdraftTransferEvaluator
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Find the repository root from the current directory or notebook path."""
    start = (start or Path.cwd()).resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find pyproject.toml in the current directory or its parents."
    )


ROOT = find_project_root()

evaluator = OverdraftTransferEvaluator(
    ROOT / "implementations/tabular_data/multi_table/data/berka2/trans.csv",
    real_data_config={"trans_sample_size": 100_000},
)

evaluator.evaluate_synthetic(
    ROOT / "implementations/tabular_data/multi_table/results2/"
    "ClavaDDPM_notebook/trans/synthetic_final/trans_synthetic.csv",
    name="sample=100k, iterations=5000",
    config={"trans_sample_size": 100_000, "iterations": 5_000},
)

evaluator.get_metrics()
```

`real_data_config` and `config` are recorded as metadata; they do not sample rows
or change model training. The evaluator uses every row in each supplied input.

## Results

`get_metrics()` returns one row per named synthetic run, with row and overdraft
counts, overdraft prevalence, average precision, ROC AUC, precision, recall, F1,
the classification threshold (default `0.5`), and configuration metadata.
Average precision and ROC AUC are `NaN` when a synthetic input has only one class.

Call `evaluate_synthetic()` again with a different `name` to compare runs; reusing
a name replaces its results. The real holdout baseline is available through
`evaluator.real_test_metrics`. These scores measure how the real-trained detector
performs on synthetic data; they are not a complete measure of synthetic data quality.


## Reusable transaction-focused report

If starting with no experiments, first run
[prepare_account_sample.ipynb](../experiments/prepare_account_sample.ipynb) to
prepare the selected account cohort, then
[00_baseline.ipynb](../experiments/00_baseline.ipynb) to train and generate a reference. Then open
[the evaluation notebook](../experiments/transaction_overdraft_evaluation.ipynb).
It discovers completed runs from the new workflow and uses one detector for all
of them; empty discovery prints the next action instead of assuming results.
From the repository root, after generation:

```python
from pathlib import Path
from implementations.tabular_data.multi_table.evaluation.transaction_evaluation import evaluate_saved_runs
from implementations.tabular_data.multi_table.experiments.tuning import (
    DEFAULT_EXPERIMENT_ROOT, DEFAULT_REPORT_ROOT, completed_runs,
)
from implementations.tabular_data.multi_table.experiments.dataset_settings import DATA_DIR

runs = completed_runs(DATA_DIR, DEFAULT_EXPERIMENT_ROOT)
if runs:
    comparison, report_dir = evaluate_saved_runs(
        DATA_DIR, runs, DEFAULT_REPORT_ROOT, bootstrap_repetitions=200,
    )
else:
    print("Create your first reference run with experiments/00_baseline.ipynb.")
```

Each run must contain trusted `synthetic_final_with_keys.pkl` with `account` and
`trans` DataFrames. The full-schema reference requires every CSV/domain declared in
`dataset_meta.json`, plus `preprocess_meta.json` with the category codebooks and date epochs. Both real and
synthetic `account_id` values are needed for grouped uncertainty and relationships.
Invalid/missing target balances are rejected rather than silently relabeled.

`transaction_quality.py` provides reusable numerical KS, categorical total
variation, mixed correlations (reusing the toolkit), overdraft-conditional
comparisons, category event rates with support, account profiles including inactive
accounts, and epoch-aligned chronology. The external counterparty number is an ID,
not a numerical feature. Original transaction IDs are retained for integrity checks.

`transaction_evaluation.py` adds account-bootstrap intervals, AP at a shared class
prevalence, per-run report files, source/settings fingerprints, and comparison
plots. It saves real train/test account IDs and fixed predictions. Correlation
pairs with constants remain undefined and are counted. Every metric has its own
column; there is no opaque combined score.

### Interpretation

- **Transfer direction:** train on real → evaluate on synthetic. This does not
  measure how well a model trained on synthetic data generalizes to new real users.
- **Target:** current `balance < 0`, not future overdraft. No loan classification.
- **Uncertainty:** 200 account-bootstrap draws by default, fixed detector. The
  report counts valid two-class replicates; fewer than 20 yield missing intervals.
  This is exploratory percentile uncertainty, not variation across generator seeds.
- **Prevalence:** report the actual real and synthetic event rates; differences
  affect AP. Standardized AP reweights classes to the real holdout prevalence, but does not
  correct every difference in data difficulty. Do not rank by raw AP alone.
- **Holdout:** accounts are separate for detector fitting/testing. Internal
  generator splits do not establish an independent relational outer test set.

The default classification threshold remains 0.5 for comparability; it has not
been chosen for a deployment precision/recall target. Any future threshold choice
needs a separate real validation set, not these test or synthetic outcomes.

The evaluator now aligns categorical float/int codes consistently on both sides,
retains missing versus unknown categories, caches probabilities for diagnostics,
and exposes `classification_metrics` without fitting another classifier. Existing
constructor and `evaluate_synthetic`/`get_metrics` APIs remain available.

Reference concepts: [scikit-learn precision–recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).


## Full multi-table diagnostics — enabled in every report

`multi_table_quality.py` extracts/reuses the earlier notebook metrics through
shared distribution helpers and the toolkit's mixed-correlation implementation.
Every report now contains **both** full-schema quality and transaction/overdraft
quality. The latter uses the same fixed real-trained detector across runs; no
loan classifier or synthetic-trained real-test utility model has been added.

| Report file per trial | Contents |
| --- | --- |
| `all_table_scores.csv` | Row counts/scaled deviations, PK validity, per-table KS/TVD, missing/unknown rates, within-table association errors. |
| `all_column_distributions.csv` | Individual column scores and support/invalid counts. |
| `all_edge_scores.csv` | Every direct edge: FK validity, child-count KS including zeros, joined-row coverage, and parent/child association errors. |
| `all_cross_table_correlations.csv` / `all_within_table_correlations.csv` | Pair-level diagnostics and undefined pairs. |
| `multi_parent_scores.csv` / `multi_parent_correlations.csv` | Client/account attributes joined through `disp`; mixed associations and cross-parent joint categorical TVD. |
| `excluded_features.csv` | Domain features excluded as identifiers (`trans.account`, `order.account_to`, and any internal IDs). |

`multi_table_quality.png` accompanies the transaction/overdraft plots. Global
numerical/categorical summaries average per-table means equally. Eligible missing
column/edge scores propagate as missing; mixed-correlation RMSE uses defined pairs
and reports undefined-pair counts alongside it. Table and edge breakdowns should
be inspected rather than relying on the summaries alone. Empty/invalid joins do
not become perfect zero-error relationships.

The additional joint categorical TVD is needed because an association-strength
statistic alone can miss a change in specific category combinations. These are
attribute-based comparisons; numerical values of synthetic IDs are not compared
with real identities. There is no combined task/fidelity winner score.

## Selecting another reference dataset

Edit `../experiments/dataset_settings.py` or override notebook `DATA_DIR` consistently.
All references are cohort-specific. Reports fingerprint every table, domain,
schema, preprocessing metadata, and available cohort manifest. A trial with a
recorded input signature that differs from the selected reference is rejected
before scoring. Historical trials without signatures can still be evaluated
explicitly, but provenance cannot be independently established from their missing
metadata. Current experiment discovery selects only complete matching inputs.

Full account histories are retained by the preparation workflow; the selected
500-account pilot is not the complete raw population. Use the actual event/account
counts and uncertainty rather than assuming a fixed number of real test positives.
