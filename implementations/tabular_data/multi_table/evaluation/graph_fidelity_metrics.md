# Metrics in `berka_graph_fidelity.py`

Graph model: nodes are accounts (`account_id`) and counterparties (`bank:account`, rows with `account == 0` have none). Each transaction with a counterparty is an edge. Edges between the same pair are repeats ("recurring payments"). No node matching between real and synthetic is needed.

## Reading the output

| Column | Meaning |
|---|---|
| `real`, `synthetic` | Summary value (mean for distribution metrics). Empty for grouped/conditional metrics. Can hide distribution differences, so rely on `distance`. |
| `distance_measure` | Name of the distance used for this metric (see below). |
| `distance_explanation` | One-sentence description of how that distance is computed. |
| `distance` | Difference between real and synthetic under that measure. 0 = identical. |
| `noise_floor` | The same distance between two random halves of the real accounts (what pure sampling noise looks like). |
| `ratio` | `distance / noise_floor` (floor clipped at 0.001). Above 2 is flagged `GAP`. Large ratios only mean "far from noise"; compare `distance` for magnitude. |

## Distance measures

| `distance_measure` | Computation | Compares the full distribution? |
|---|---|---|
| KS statistic | Largest gap between the two cumulative distribution curves. Range 0 to 1. | Yes. |
| Total variation | Half the sum of absolute differences between category shares. Range 0 to 1. | Yes (categorical). |
| Absolute difference | `abs(real - synthetic)` of one number. | No: the metric is a single number (rate, share, z-score). |
| Absolute log-ratio | `abs(log((real + eps) / (synthetic + eps)))` for an unbounded count. | No: single number, but scale-free. |
| Mean absolute matrix difference | Mean over all cells of `abs(real - synthetic)` of the bank co-usage matrix. | Yes, for the pairwise structure, cell by cell. |
| Weighted absolute rate difference | Per category `abs(real rate - synthetic rate)`, averaged with real category size as weights. | Yes: a rate is the full distribution of a yes/no variable. |
| KS statistic / Total variation per bucket, weighted | KS (numeric) or total variation (categorical) within each graph-position bucket, averaged with real bucket size as weights; a bucket missing in synthetic counts as 1. | Yes, within each bucket. |
| Classifier AUC distance | `abs(AUC - 0.5)` of a classifier separating real from synthetic accounts using all per-account features together. | Yes, jointly across features. |

Repeat-dependent metrics (`edge_consistency`, `temporal`) need at least 30 repeated edges/gaps on a side; with fewer the side counts as "absent" and the distance is 1.0.

## presence

| Metric | What it measures |
|---|---|
| `cp_presence_rate` | Share of all transactions that have a counterparty. |
| `accounts_with_any_cp` | Share of accounts with at least one counterparty transaction. |
| `cp_share_per_account` | Distribution (KS) of each account's share of counterparty transactions. |
| `cp_presence_by_type` / `_operation` / `_k_symbol` | Whether counterparty presence depends on the category as in the real data (weighted absolute difference of rates per category). |

## account (per-account features)

| Metric | What it measures |
|---|---|
| `n_tx` | Distribution of transactions per account. |
| `n_cp_tx` | Distribution of counterparty transactions per account. |
| `n_distinct_cp` | Distribution of distinct counterparties per account. |
| `repeat_rate` | Per account: 1 - distinct counterparties / counterparty transactions. |
| `cp_entropy` | Per account entropy of how payments are spread over counterparties (low = concentrated). |
| `frac_shared_cp` | Per account: share of its counterparties that other accounts also pay. |

## pair

| Metric | What it measures |
|---|---|
| `edge_multiplicity` | Distribution of the number of transactions per (account, counterparty) pair. |
| `repeat_tx_share` | Share of counterparty transactions that repeat an earlier pair. |

## counterparty

| Metric | What it measures |
|---|---|
| `cp_indegree` | Distribution of distinct accounts paying each counterparty. |
| `cp_tx_count` | Distribution of transactions received by each counterparty. |
| `tx_to_shared_cp_share` | Share of counterparty transactions that go to a counterparty used by 2+ accounts. |
| `top1pct_cp_tx_share` | Share of counterparty transactions going to the top 1% of counterparties (hub concentration). |
| `acct_number_multi_bank_share` | Share of counterparty account numbers that appear under more than one bank (should be ~0). |

## edge_consistency (repeated pairs only)

| Metric | What it measures |
|---|---|
| `repeated_edge_k_symbol_purity` | Share of repeated pairs whose `k_symbol` never changes. |
| `repeated_edge_operation_purity` | Same for `operation`. |
| `repeated_edge_type_purity` | Same for `trans_type`. |
| `repeated_edge_log_amount_sd` | Distribution (KS) of the within-pair spread of log-amount. |

## structure

| Metric | What it measures |
|---|---|
| `giant_component_frac` | Share of all nodes in the largest connected component. |
| `dyad_component_share` | Share of components that are a single account-counterparty pair. |
| `wedges_per_edge`, `butterflies_per_edge` | Two-step paths and 4-cycles (account-cp-account-cp) per edge. Only reported if at least 5% of real counterparties are shared (`min_shared_counterparty_share_for_motifs`). |
| `butterfly_excess_vs_null`, `butterfly_z_vs_null` | Observed butterflies relative to a degree-preserving edge-swap null (same gating). |

## bank

| Metric | What it measures |
|---|---|
| `bank_mix` | Total variation between the bank distribution of counterparty transactions. |
| `banks_per_account` | Distribution of distinct banks used per account. |
| `bank_cousage_matrix` | Mean absolute difference of the matrix of "share of accounts using both bank i and bank j". |

## temporal (repeated pairs only)

| Metric | What it measures |
|---|---|
| `gap_days` | Distribution of days between consecutive transactions of the same pair. |
| `gap_cv_per_pair` | Distribution of the regularity (std/mean of gaps) per pair; low = periodic. |
| `monthly_gap_share` | Share of gaps that are within 3 days of a whole number of months. |
| `date_parse_fail_rate` | Share of rows whose date cannot be parsed (sanity check). |

## coupling (attribute mix conditional on graph position)

Each row is the weighted distance between the real and synthetic distribution of a variable within buckets, weighted by real bucket size. A bucket present in the real data but missing in the synthetic data counts as distance 1.

| Metric | What it measures |
|---|---|
| `log_amt`, `type`, `operation`, `k_symbol`, `bank` `\| edge_mult bucket` | Variable distribution within pair-multiplicity buckets (1, 2-3, 4+). |
| same five `\| cp_indeg bucket` | Variable distribution within counterparty in-degree buckets (1, 2+). |

## c2st

| Metric | What it measures |
|---|---|
| `account_graph_features_AUC` | A gradient-boosted classifier tries to tell real from synthetic accounts using the per-account graph features above. 0.5 = indistinguishable, 1.0 = perfectly separable. The distance is `abs(AUC - 0.5)`; the floor comes from real halves. |
