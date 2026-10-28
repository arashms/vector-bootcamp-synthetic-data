# Overdraft transfer evaluation

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

Open [the evaluation notebook](../experiments/transaction_overdraft_evaluation.ipynb)
for the complete saved-run analysis. It uses this same detector once for all runs.
From the repository root:

```python
from pathlib import Path
from implementations.tabular_data.multi_table.evaluation.transaction_evaluation import evaluate_saved_runs

root = Path("implementations/tabular_data/multi_table")
runs = [root / "results/experiments/guidance_strength_first_pass" / f"guidance_{s}_seed42"
        for s in [0, 0.5, 1, 2, 4]]
comparison, report_dir = evaluate_saved_runs(
    root / "data/berka", runs, root / "results/experiments/transaction_overdraft",
    bootstrap_repetitions=200,
)
```

Each run must contain trusted `synthetic_final_with_keys.pkl` with `account` and
`trans` DataFrames. The reference requires the matching CSVs and
`preprocess_meta.json` with the category codebooks and date epochs. Both real and
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
- **Prevalence:** synthetic runs have more overdrafts than the full real reference.
  Standardized AP reweights classes to the real holdout prevalence, but does not
  correct every difference in data difficulty. Do not rank by raw AP alone.
- **Holdout:** accounts are separate for detector fitting/testing. Existing
  generator checkpoints do not establish an independent relational outer test set.

The default classification threshold remains 0.5 for comparability; it has not
been chosen for a deployment precision/recall target. Any future threshold choice
needs a separate real validation set, not these test or synthetic outcomes.

The evaluator now aligns categorical float/int codes consistently on both sides,
retains missing versus unknown categories, caches probabilities for diagnostics,
and exposes `classification_metrics` without fitting another classifier. Existing
constructor and `evaluate_synthetic`/`get_metrics` APIs remain available.

Reference concepts: [scikit-learn precision–recall](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)
and [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html).
