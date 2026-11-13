"""Rule-based overdraft-episode post-processing of saved synthetic samples (CPU).

Experimental, not learned by the generator. Scattered synthetic negative balances
are made positive, then overdraft episodes are placed in a real-like share of
accounts. Calibration uses only summary statistics of the real *train* accounts.

Variants:
- ``episodes``: only balances change (concentrates the overdraft label).
- ``episodes_profile``: episode rows also get categorical fields drawn from the
  joint frequency of real negative rows and amounts from a log-normal fit.

Usage (CPU only)::

    python -m <module path>.experiments.analysis.overdraft_episodes \
        REPORT_DIR OUT_ROOT --runs RUN [RUN ...] [--seed 0]

REPORT_DIR provides ``comparison_with_settings.csv`` (runs to process) and
``real_account_split.csv`` (train accounts). Each run is copied to
``OUT_ROOT/<variant>/<run>`` with a modified ``synthetic_final_with_keys.pkl``.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORICAL = ["trans_type", "operation", "k_symbol"]
VARIANTS = ("episodes", "episodes_profile")


def calibrate(real_trans: pd.DataFrame, train_accounts) -> dict:
    """Return episode statistics from real train accounts only."""
    t = real_trans[real_trans.account_id.isin(train_accounts)]
    t = t.sort_values(["account_id", "trans_date", "trans_id"]).assign(neg=lambda f: f.balance < 0)
    overdrawn = t.groupby("account_id").neg.any()
    g = t[t.account_id.isin(overdrawn[overdrawn].index)].copy()
    g["run"] = (g.neg != g.groupby("account_id").neg.shift()).cumsum()
    neg = g[g.neg]
    log_balance = np.log(-neg.balance)
    log_amount = np.log1p(neg.amount.abs())
    return {
        "account_rate": float(overdrawn.mean()),
        "episodes_per_account": neg.groupby("account_id").run.nunique().tolist(),
        "episode_lengths": neg.groupby("run").size().tolist(),
        "log_balance": (float(log_balance.mean()), float(log_balance.std())),
        "log_amount": (float(log_amount.mean()), float(log_amount.std())),
        "profile": neg[CATEGORICAL].value_counts(normalize=True),
    }


def postprocess(trans: pd.DataFrame, stats: dict, variant: str, rng) -> pd.DataFrame:
    """Return a copy of ``trans`` with overdraft episodes placed per account."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}.")
    t = trans.copy()
    t["balance"] = pd.to_numeric(t.balance).abs()
    t["amount"] = pd.to_numeric(t.amount)
    order = t.sort_values(["account_id", "trans_date"]).groupby("account_id").indices
    accounts = [a for a, rows in order.items() if len(rows) >= 2]
    chosen = rng.choice(len(accounts), size=round(stats["account_rate"] * len(accounts)), replace=False)
    positions = t.sort_values(["account_id", "trans_date"]).index.to_numpy()
    selected = []
    for i in chosen:
        rows = positions[order[accounts[i]]]
        for _ in range(rng.choice(stats["episodes_per_account"])):
            length = min(int(rng.choice(stats["episode_lengths"])), len(rows))
            start = rng.integers(0, len(rows) - length + 1)
            selected.extend(rows[start : start + length])
    selected = pd.Index(selected).unique()
    mu, sigma = stats["log_balance"]
    t.loc[selected, "balance"] = -np.exp(rng.normal(mu, sigma, len(selected)))
    if variant == "episodes_profile":
        profile = stats["profile"]
        picks = rng.choice(len(profile), size=len(selected), p=profile.to_numpy())
        for j, column in enumerate(CATEGORICAL):
            values = np.array([profile.index[k][j] for k in picks], dtype=float)
            t.loc[selected, column] = values
        mu, sigma = stats["log_amount"]
        t.loc[selected, "amount"] = np.expm1(rng.normal(mu, sigma, len(selected))).round(2)
    return t


def main():
    """Write post-processed copies of the selected runs."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("out_root", type=Path)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("implementations/tabular_data/multi_table/data/berka_accounts_500_seed42"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", nargs="+", required=True, help="Run directory names to process.")
    args = parser.parse_args()

    comparison = pd.read_csv(args.report_dir / "comparison_with_settings.csv", index_col=0)
    comparison = comparison.loc[args.runs]
    split = pd.read_csv(args.report_dir / "real_account_split.csv")
    stats = calibrate(pd.read_csv(args.data_dir / "trans.csv"), split.loc[split.split == "train", "account_id"])
    print(f"Calibrated on train accounts: overdrawn rate {stats['account_rate']:.3f}")
    for run in comparison.run_directory:
        source = Path(run)
        with (source / "synthetic_final_with_keys.pkl").open("rb") as stream:
            synthetic = pickle.load(stream)
        for variant in VARIANTS:
            target = args.out_root / variant / f"{source.name}_{variant}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("config.yaml", "manifest.json"):
                shutil.copy2(source / name, target / name)
            rng = np.random.default_rng(args.seed)
            out = {**synthetic, "trans": postprocess(synthetic["trans"], stats, variant, rng)}
            with (target / "synthetic_final_with_keys.pkl").open("wb") as stream:
                pickle.dump(out, stream)
            (target / "postprocess.json").write_text(
                json.dumps({"source": str(source), "variant": variant, "seed": args.seed}, indent=2),
            )
            print(f"Wrote {target}")


if __name__ == "__main__":
    main()
