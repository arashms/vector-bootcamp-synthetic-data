"""Train-on-synthetic, test-on-real overdraft detection (CPU).

Run from the repo root (module path under implementations.tabular_data.multi_table):
    PYTHONPATH=. python -m <module path>.experiments.analysis.tstr_overdraft

Requires the step-10 sample runs and the `overdraft_episodes.py` variants.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from implementations.tabular_data.multi_table.evaluation.overdraft_transfer_evaluator import (
    DEFAULT_RANDOM_FOREST_PARAMS,
    FEATURES,
    TARGET,
    OverdraftTransferEvaluator,
)
from implementations.tabular_data.multi_table.experiments.dataset_settings import DATA_DIR
from implementations.tabular_data.multi_table.experiments.tuning import DEFAULT_EXPERIMENT_ROOT


# Step-10 final runs, training seeds 42-46.
runs = [
    "db56cca1027418cfb4bd",
    "ffcf4e23ea220d4f65c7",
    "3a0f2481e5e0cad98e5f",
    "f56ae98fdfd44e788b1a",
    "0206fdd9c07b79be9bb0",
]
ep = Path("implementations/tabular_data/multi_table/results/experiments/overdraft_episodes")
ev = OverdraftTransferEvaluator(pd.read_csv(Path(DATA_DIR) / "trans.csv"), random_state=42)
test = ev.real_test
y = test[TARGET].to_numpy()
rows = [
    {
        "variant": "real_train",
        "seed": "-",
        "roc_auc": roc_auc_score(y, ev.real_test_probabilities),
        "ap": average_precision_score(y, ev.real_test_probabilities),
        "train_pos": int(ev.real_train[TARGET].sum()),
    },
]
dirs = {"original": [DEFAULT_EXPERIMENT_ROOT / "samples" / r for r in runs]}
dirs |= {v: [ep / v / f"{r}_{v}" for r in runs] for v in ("episodes", "episodes_profile")}
for variant, paths in dirs.items():
    for seed, path in zip(range(42, 47), paths):
        with (path / "synthetic_final_with_keys.pkl").open("rb") as f:
            syn = ev._prepare_data(pickle.load(f)["trans"], path.name, require_accounts=False)
        model = ev._build_model(DEFAULT_RANDOM_FOREST_PARAMS.copy())
        model.fit(syn[FEATURES], syn[TARGET])
        p = model.predict_proba(test[FEATURES])[:, 1]
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "roc_auc": roc_auc_score(y, p),
                "ap": average_precision_score(y, p),
                "train_pos": int(syn[TARGET].sum()),
            },
        )
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
print("real test prevalence", y.mean(), "rows", len(y))
print(df.to_string(index=False))
print(df.groupby("variant")[["roc_auc", "ap"]].agg(["mean", "std"]).round(4))
out = ep / "tstr"
out.mkdir(exist_ok=True)
df.to_csv(out / "tstr.csv", index=False)
(out / "README.json").write_text(
    json.dumps({"task": "train RF on synthetic trans, test on real held-out accounts (split seed 42)"}),
)
