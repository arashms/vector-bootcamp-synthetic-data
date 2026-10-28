"""Reusable transaction fidelity and account-level diagnostics for Berka."""

# Toolkit import is deferred until mixed correlations are requested.
# ruff: noqa: PLC0415
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from implementations.tabular_data.multi_table.evaluation.overdraft_transfer_evaluator import (
    CATEGORICAL_FEATURES,
    align_categories,
)


TRANSACTION_NUMERICAL = ["trans_date", "amount", "balance"]


def distribution_differences(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical: Sequence[str],
    categorical: Sequence[str],
    category_reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare columns with KS/TVD, retaining missingness and sample counts.

    Numerical distances use finite observations; missing/invalid rates are
    reported separately. Categorical missing and unknown values are retained.
    Identifiers must not be passed as numerical measurements.
    """
    rows = []
    for column in [*numerical, *categorical]:
        original, generated = real[column], synthetic[column]
        if column in numerical:
            r = pd.to_numeric(original, errors="coerce")
            s = pd.to_numeric(generated, errors="coerce")
            valid_r, valid_s = np.isfinite(r), np.isfinite(s)
            distance = ks_2samp(r[valid_r], s[valid_s]).statistic if valid_r.any() and valid_s.any() else np.nan
            kind = "ks"
            unknown_rate = np.nan
        else:
            reference = original if category_reference is None else category_reference[column]
            r, s = align_categories(original, reference), align_categories(generated, reference)
            valid_r, valid_s = original.notna(), generated.notna()
            r_freq, s_freq = r.value_counts(normalize=True).align(s.value_counts(normalize=True), fill_value=0)
            distance = float((r_freq - s_freq).abs().sum() / 2) if len(r) and len(s) else np.nan
            kind = "tvd"
            unknown_rate = float(s.eq("__unknown__").mean())
        rows.append(
            {
                "column": column,
                "metric": kind,
                "distance": float(distance),
                "real_n": len(original),
                "synthetic_n": len(generated),
                "real_invalid_rate": float((~valid_r).mean()),
                "synthetic_invalid_rate": float((~valid_s).mean()),
                "synthetic_unknown_rate": unknown_rate,
            },
        )
    return pd.DataFrame(rows)


def correlation_differences(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical: list[str],
    categorical: list[str],
    parent_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Reuse the toolkit mixed-correlation implementation and expose each pair.

    With parent_columns, retain only parent-by-child pairs. Undefined or constant
    pairs remain missing; the result reports how many pairs can be compared.
    """
    from midst_toolkit.evaluation.quality import CorrelationMatrixDifference

    from implementations.tabular_data.evaluation.preprocessing import syntheval_preprocess

    columns = numerical + categorical
    r, s = real[columns].copy(), synthetic[columns].copy()
    for column in numerical:
        r[column] = pd.to_numeric(r[column], errors="coerce")
        s[column] = pd.to_numeric(s[column], errors="coerce")
    for column in categorical:
        r[column] = align_categories(real[column], real[column])
        s[column] = align_categories(synthetic[column], real[column])
    pairs = [
        (left, right)
        for i, left in enumerate(columns)
        for right in columns[i + 1 :]
        if parent_columns is None or ((left in parent_columns) != (right in parent_columns))
    ]
    usable = [c for c in columns if r[c].nunique() > 1 and s[c].nunique() > 1]
    rows = []
    matrices = None
    if usable and len(r) > 1 and len(s) > 1:
        num = [c for c in numerical if c in usable]
        cat = [c for c in categorical if c in usable]
        r_clean = r[usable].replace([np.inf, -np.inf], np.nan).dropna()
        s_clean = s[usable].replace([np.inf, -np.inf], np.nan).dropna()
        if len(r_clean) > 1 and len(s_clean) > 1:
            r_enc, s_enc = syntheval_preprocess(num, cat, r_clean, s_clean)
            metric = CorrelationMatrixDifference(cat, num, do_preprocess=False, compute_mixed_correlations=True)
            metric.compute(r_enc, s_enc)
            matrices = metric.syntheval_metric.evaluate(mixed_corr=True, return_mats=True)
    for left, right in pairs:
        r_corr = s_corr = np.nan
        if matrices is not None and left in usable and right in usable:
            r_corr = float(matrices["real_cor_mat"].loc[left, right])
            s_corr = float(matrices["synt_cor_mat"].loc[left, right])
        rows.append(
            {"left": left, "right": right, "real": r_corr, "synthetic": s_corr, "absolute_error": abs(r_corr - s_corr)},
        )
    return pd.DataFrame(rows, columns=["left", "right", "real", "synthetic", "absolute_error"])


def overdraft_group_differences(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Compare category-specific event rates and show their support counts."""
    rows = []
    for column in CATEGORICAL_FEATURES:
        grouped = []
        for frame in (real, synthetic):
            values = align_categories(frame[column], real[column])
            grouped.append(
                pd.DataFrame({"category": values, "overdraft": frame.balance.lt(0).to_numpy()})
                .groupby("category")
                .overdraft.agg(["count", "sum", "mean"]),
            )
        both = grouped[0].add_prefix("real_").join(grouped[1].add_prefix("synthetic_"), how="outer")
        both["rate_error"] = (both.real_mean - both.synthetic_mean).abs()
        for category, record in both.iterrows():
            rows.append({"column": column, "category": category, **record.to_dict()})
    return pd.DataFrame(rows)


def account_profiles(accounts: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Count observed transactions and overdrafts, including inactive accounts."""
    if accounts.account_id.isna().any() or accounts.account_id.duplicated().any():
        raise ValueError("Account primary keys must be present and unique for account profiles.")
    trans = transactions.copy()
    trans["account_id"] = pd.to_numeric(trans.account_id, errors="raise")
    trans["overdraft"] = pd.to_numeric(trans.balance, errors="coerce").lt(0).astype(int)
    trans["amount"] = pd.to_numeric(trans.amount, errors="coerce")
    grouped = trans.groupby("account_id").agg(
        transactions=("overdraft", "size"),
        overdrafts=("overdraft", "sum"),
        total_amount=("amount", "sum"),
    )
    parent_ids = pd.Index(pd.to_numeric(accounts.account_id, errors="raise"), name="account_id")
    profile = grouped.reindex(parent_ids, fill_value=0)
    profile["ever_overdrawn"] = profile.overdrafts.gt(0).astype(int)
    return profile


def transaction_quality(
    real: dict[str, pd.DataFrame],
    synthetic: dict[str, pd.DataFrame],
    date_epochs: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Evaluate transaction marginals, target relationships, and account links."""
    r, s = real["trans"].copy(), synthetic["trans"].copy()
    for frame in (r, s):
        for column in TRANSACTION_NUMERICAL:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    distributions = distribution_differences(r, s, TRANSACTION_NUMERICAL, CATEGORICAL_FEATURES)
    correlations = correlation_differences(r, s, TRANSACTION_NUMERICAL, CATEGORICAL_FEATURES)
    groups = overdraft_group_differences(r, s)
    conditional = []
    for label in (0, 1):
        subset_r, subset_s = r[r.balance.lt(0) == bool(label)], s[s.balance.lt(0) == bool(label)]
        scores = distribution_differences(subset_r, subset_s, TRANSACTION_NUMERICAL, CATEGORICAL_FEATURES, r)
        scores["overdraft"] = label
        conditional.append(scores)
    profiles = [account_profiles(data["account"], data["trans"]) for data in (real, synthetic)]
    account_scores = distribution_differences(
        profiles[0], profiles[1], ["transactions", "overdrafts", "total_amount"], [],
    )
    valid_fk = pd.to_numeric(s.account_id, errors="coerce").isin(pd.to_numeric(synthetic["account"].account_id))
    summary = {
        "transaction_ks": distributions.loc[distributions.metric == "ks", "distance"].mean(skipna=False),
        "transaction_tvd": distributions.loc[distributions.metric == "tvd", "distance"].mean(skipna=False),
        "transaction_correlation_rmse": np.sqrt((correlations.absolute_error**2).mean()),
        "undefined_correlation_pairs": int(correlations.absolute_error.isna().sum()),
        "transaction_fk_valid_rate": valid_fk.mean(),
        "transaction_pk_unique_rate": s.trans_id.nunique() / len(s) if len(s) else np.nan,
        "transactions_per_account_ks": account_scores.loc[account_scores.column == "transactions", "distance"].iloc[0],
        "real_accounts_overdrawn_rate": profiles[0].ever_overdrawn.mean(),
        "synthetic_accounts_overdrawn_rate": profiles[1].ever_overdrawn.mean(),
        "max_invalid_rate": distributions.synthetic_invalid_rate.max(),
        "max_unknown_category_rate": distributions.synthetic_unknown_rate.max(),
    }
    summary["accounts_overdrawn_rate_gap"] = abs(
        summary["synthetic_accounts_overdrawn_rate"] - summary["real_accounts_overdrawn_rate"],
    )
    positive = conditional[1]
    summary["overdraft_numerical_ks"] = positive.loc[positive.metric == "ks", "distance"].mean(skipna=False)
    summary["overdraft_categorical_tvd"] = positive.loc[positive.metric == "tvd", "distance"].mean(skipna=False)
    joined = []
    for source in (real, synthetic):
        trans = source["trans"][["account_id", *TRANSACTION_NUMERICAL, *CATEGORICAL_FEATURES]].copy()
        accounts = source["account"][["account_id", "account_date", "frequency"]].copy()
        trans["account_id"] = pd.to_numeric(trans.account_id)
        accounts["account_id"] = pd.to_numeric(accounts.account_id)
        joined.append(trans.merge(accounts, on="account_id", validate="many_to_one"))
    account_pairs = correlation_differences(
        *joined,
        ["account_date", *TRANSACTION_NUMERICAL],
        ["frequency", *CATEGORICAL_FEATURES],
        parent_columns=["account_date", "frequency"],
    )
    summary["account_transaction_correlation_rmse"] = np.sqrt((account_pairs.absolute_error**2).mean())
    summary["undefined_account_pairs"] = int(account_pairs.absolute_error.isna().sum())
    periods = []
    if date_epochs and date_epochs.get("trans") and date_epochs.get("account"):
        trans_epoch = pd.to_datetime(date_epochs["trans"], format="%y%m%d")
        account_epoch = pd.to_datetime(date_epochs["account"], format="%y%m%d")
        for label, frame in zip(("real", "synthetic"), joined, strict=True):
            transaction_date = trans_epoch + pd.to_timedelta(pd.to_numeric(frame.trans_date), unit="D")
            opening_date = account_epoch + pd.to_timedelta(pd.to_numeric(frame.account_date), unit="D")
            summary[f"{label}_transaction_before_account_rate"] = float((transaction_date < opening_date).mean())
            monthly = pd.DataFrame(
                {
                    "month": transaction_date.dt.to_period("M").astype(str),
                    "overdraft": pd.to_numeric(frame.balance).lt(0).astype(int),
                },
            )
            monthly = monthly.groupby("month").overdraft.agg(["count", "sum", "mean"]).reset_index()
            monthly["dataset"] = label
            periods.append(monthly)
    details = {
        "distributions": distributions,
        "correlations": correlations,
        "overdraft_by_category": groups,
        "conditional_distributions": pd.concat(conditional, ignore_index=True),
        "account_distributions": account_scores,
        "account_correlations": account_pairs,
        "real_account_profiles": profiles[0].reset_index(),
        "synthetic_account_profiles": profiles[1].reset_index(),
        "monthly_overdraft": pd.concat(periods, ignore_index=True) if periods else pd.DataFrame(),
    }
    return summary, details
