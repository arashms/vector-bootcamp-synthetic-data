"""Full-schema quality diagnostics shared by every hyperparameter experiment.

Reuse the transaction distribution helpers and toolkit mixed correlations.
Report individual tables/relationships; never collapse task and fidelity scores
into an automatic winner. Identifiers are used for integrity and joins only.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from implementations.tabular_data.multi_table.evaluation.overdraft_transfer_evaluator import align_categories
from implementations.tabular_data.multi_table.evaluation.transaction_quality import (
    correlation_differences,
    distribution_differences,
)


NON_MEASUREMENT_COLUMNS = {"trans": {"account"}, "order": {"account_to"}}


def feature_columns(table, domain):
    """Use domain types while excluding internal and external identifiers."""
    numerical, categorical, excluded = [], [], []
    for column, spec in domain.items():
        if "_id" in column or column in NON_MEASUREMENT_COLUMNS.get(table, set()):
            excluded.append({"table": table, "column": column, "reason": "identifier"})
        elif spec["type"] == "continuous":
            numerical.append(column)
        elif spec["type"] == "discrete":
            categorical.append(column)
        else:
            raise ValueError(f"Unsupported domain type: {table}.{column} = {spec['type']}")
    return numerical, categorical, excluded


def _valid_pk(frame, table):
    key = pd.to_numeric(frame[f"{table}_id"], errors="coerce")
    return bool(len(key) and np.isfinite(key).all() and not key.duplicated().any())


def _rmse(frame):
    return float(np.sqrt(frame.absolute_error.pow(2).mean())) if len(frame) else np.nan


def _joined_features(source, anchor, feature_tables, domains):
    parents = [table for table in feature_tables if table != anchor]
    if any(not _valid_pk(source[table], table) for table in parents):
        return None
    keys = [f"{table}_id" for table in parents]
    frame = source[anchor][keys].copy()
    for key in keys:
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    if anchor in feature_tables:
        num, cat, _ = feature_columns(anchor, domains[anchor])
        for column in num + cat:
            frame[f"{anchor}__{column}"] = source[anchor][column].to_numpy()
    for table in feature_tables:
        numerical, categorical, _ = feature_columns(table, domains[table])
        columns = numerical + categorical
        rename = {column: f"{table}__{column}" for column in columns}
        if table != anchor:
            key = f"{table}_id"
            parent = source[table][[key, *columns]].rename(columns=rename).copy()
            parent[key] = pd.to_numeric(parent[key], errors="coerce")
            frame = frame.merge(parent, on=key, how="inner", validate="many_to_one")
    return frame


def _relationship_scores(real, synthetic, domains, *, anchor, left, right):
    numerical, categorical, left_columns = [], [], []
    for table in (left, right):
        num, cat, _ = feature_columns(table, domains[table])
        prefixed_num = [f"{table}__{column}" for column in num]
        prefixed_cat = [f"{table}__{column}" for column in cat]
        numerical += prefixed_num
        categorical += prefixed_cat
        if table == left:
            left_columns = prefixed_num + prefixed_cat
    joined = [_joined_features(source, anchor, [left, right], domains) for source in (real, synthetic)]
    valid_join = all(frame is not None for frame in joined)
    frames = [frame if frame is not None else pd.DataFrame(columns=numerical + categorical) for frame in joined]
    pairs = correlation_differences(*frames, numerical, categorical, parent_columns=left_columns)
    joint_distances = []
    if anchor not in (left, right):
        for pair in pairs.itertuples():
            distance = np.nan
            if pair.left in categorical and pair.right in categorical:
                if len(frames[0]) and len(frames[1]):
                    joint = []
                    for frame in frames:
                        first = align_categories(frame[pair.left], frames[0][pair.left])
                        second = align_categories(frame[pair.right], frames[0][pair.right])
                        joint.append(pd.Series(list(zip(first, second))).value_counts(normalize=True))
                    r_freq, s_freq = joint[0].align(joint[1], fill_value=0)
                    distance = float((r_freq - s_freq).abs().sum() / 2)
                joint_distances.append(distance)
            pairs.loc[pair.Index, "joint_categorical_tvd"] = distance
    summary = {"correlation_rmse": _rmse(pairs), "pairs": len(pairs),
               "undefined_pairs": int(pairs.absolute_error.isna().sum()),
               "real_joined_rows": len(frames[0]), "synthetic_joined_rows": len(frames[1]),
               "join_status": "valid" if valid_join else "invalid_parent_primary_key"}
    if anchor not in (left, right):
        summary["joint_categorical_tvd"] = pd.Series(joint_distances, dtype=float).mean(skipna=False)
    return summary, pairs


def _eligible_mean(tables, score, count_column):
    return float(tables.loc[tables[count_column] > 0, score].mean(skipna=False))


def multi_table_quality(real, synthetic, dataset_meta, domains, *, sample_scale=1.0):
    """Score all declared tables, every direct edge, and multi-parent joins.

    Numerical/categorical summary distances average table means equally, rather
    than letting the large transaction table dominate. Eligible missing table/edge
    scores propagate as missing. Correlation RMSE uses defined pairs and always
    reports the undefined-pair count beside it.
    """
    names = list(dataset_meta["tables"])
    for label, source in (("real", real), ("synthetic", synthetic)):
        missing = set(names) - set(source)
        if missing:
            raise ValueError(f"{label} tables missing from declared schema: {sorted(missing)}")
    column_rows, table_rows, within_rows, excluded = [], [], [], []
    for table in names:
        original, generated = real[table], synthetic[table]
        if not _valid_pk(original, table):
            raise ValueError(f"Real {table} primary keys must be finite, present, and unique.")
        num, cat, omitted = feature_columns(table, domains[table])
        excluded.extend(omitted)
        columns = distribution_differences(original, generated, num, cat)
        columns["table"] = table
        column_rows.append(columns)
        within = correlation_differences(original, generated, num, cat)
        within["table"] = table
        within_rows.append(within)
        key = pd.to_numeric(generated[f"{table}_id"], errors="coerce")
        expected_rows = len(original) * sample_scale
        table_rows.append({
            "table": table, "real_rows": len(original), "synthetic_rows": len(generated),
            "expected_rows_at_scale": expected_rows,
            "relative_row_count_error": abs(len(generated) - expected_rows) / expected_rows if expected_rows else np.nan,
            "pk_unique_rate": float(key[np.isfinite(key)].nunique() / len(key)) if len(key) else np.nan,
            "pk_missing_or_invalid_rate": float((~np.isfinite(key)).mean()),
            "numerical_columns": len(num), "categorical_columns": len(cat),
            "numerical_ks": columns.loc[columns.metric == "ks", "distance"].mean(skipna=False),
            "categorical_tvd": columns.loc[columns.metric == "tvd", "distance"].mean(skipna=False),
            "max_invalid_rate": columns.synthetic_invalid_rate.max(),
            "max_unknown_category_rate": columns.synthetic_unknown_rate.max(),
            "within_table_correlation_rmse": _rmse(within), "correlation_pairs": len(within),
            "undefined_pairs": int(within.absolute_error.isna().sum()),
        })
    edge_rows, edge_pairs = [], []
    for parent, child in dataset_meta["relation_order"]:
        if parent is None:
            continue
        key = f"{parent}_id"
        counts = []
        for source in (real, synthetic):
            child_key = pd.to_numeric(source[child][key], errors="coerce")
            parent_key = pd.to_numeric(source[parent][key], errors="coerce")
            counts.append(parent_key.map(child_key.value_counts()).fillna(0))
        child_keys = pd.to_numeric(synthetic[child][key], errors="coerce")
        parent_keys = pd.to_numeric(synthetic[parent][key], errors="coerce")
        valid = np.isfinite(child_keys) & child_keys.isin(parent_keys[np.isfinite(parent_keys)])
        pair_summary, pairs = _relationship_scores(real, synthetic, domains, anchor=child, left=parent, right=child)
        pairs["edge"] = f"{parent}->{child}"
        edge_pairs.append(pairs)
        edge_rows.append({
            "edge": f"{parent}->{child}", "fk_valid_rate": float(valid.mean()),
            "invalid_foreign_keys": int((~valid).sum()), "synthetic_child_rows": len(valid),
            "cardinality_ks": float(ks_2samp(*counts).statistic)
            if len(counts[0]) and len(counts[1]) and _valid_pk(synthetic[parent], parent) else np.nan,
            "real_mean_children": float(counts[0].mean()), "synthetic_mean_children": float(counts[1].mean()),
            "real_fraction_zero_children": float(counts[0].eq(0).mean()),
            "synthetic_fraction_zero_children": float(counts[1].eq(0).mean()), **pair_summary,
        })
    multi_rows, multi_pairs = [], []
    for child, spec in dataset_meta["tables"].items():
        for left, right in combinations(spec["parents"], 2):
            scores, pairs = _relationship_scores(real, synthetic, domains, anchor=child, left=left, right=right)
            label = f"{left}->{child}<-{right}"
            pairs["relationship"] = label
            multi_pairs.append(pairs)
            multi_rows.append({"relationship": label, **scores})
    tables = pd.DataFrame(table_rows)
    edges = pd.DataFrame(edge_rows)
    multiparent = pd.DataFrame(multi_rows, columns=[
        "relationship", "correlation_rmse", "pairs", "undefined_pairs", "real_joined_rows", "synthetic_joined_rows", "join_status", "joint_categorical_tvd",
    ])
    summary = {
        "all_table_count": len(tables), "all_edge_count": len(edges),
        "all_table_numerical_ks": _eligible_mean(tables, "numerical_ks", "numerical_columns"),
        "all_table_categorical_tvd": _eligible_mean(tables, "categorical_tvd", "categorical_columns"),
        "all_within_table_correlation_rmse": _eligible_mean(tables, "within_table_correlation_rmse", "correlation_pairs"),
        "all_relationship_correlation_rmse": edges.correlation_rmse.mean(skipna=False) if len(edges) else np.nan,
        "all_cardinality_ks": edges.cardinality_ks.mean(skipna=False) if len(edges) else np.nan,
        "all_min_fk_valid_rate": edges.fk_valid_rate.min(skipna=False) if len(edges) else np.nan,
        "all_min_pk_unique_rate": tables.pk_unique_rate.min(skipna=False),
        "all_max_invalid_rate": tables.max_invalid_rate.max(),
        "all_max_unknown_category_rate": tables.max_unknown_category_rate.max(),
        "all_undefined_distribution_columns": int(sum(frame.distance.isna().sum() for frame in column_rows)),
        "all_undefined_correlation_pairs": int(tables.undefined_pairs.sum() + edges.undefined_pairs.sum()) if len(edges)
        else int(tables.undefined_pairs.sum()),
        "multi_parent_correlation_rmse": multiparent.correlation_rmse.mean(skipna=False) if len(multiparent) else np.nan,
        "multi_parent_undefined_pairs": int(multiparent.undefined_pairs.sum()),
        "multi_parent_joint_categorical_tvd": multiparent.joint_categorical_tvd.mean(skipna=False) if len(multiparent) else np.nan,
    }
    details = {
        "all_table_scores": tables, "all_column_distributions": pd.concat(column_rows, ignore_index=True),
        "all_within_table_correlations": pd.concat([frame for frame in within_rows if len(frame)], ignore_index=True)
        if any(len(frame) for frame in within_rows) else pd.DataFrame(),
        "all_edge_scores": edges,
        "all_cross_table_correlations": pd.concat(edge_pairs, ignore_index=True) if edge_pairs else pd.DataFrame(),
        "multi_parent_scores": multiparent,
        "multi_parent_correlations": pd.concat(multi_pairs, ignore_index=True) if multi_pairs else pd.DataFrame(),
        "excluded_features": pd.DataFrame(excluded, columns=["table", "column", "reason"]),
    }
    return summary, details
