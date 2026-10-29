# Multi-table synthetic data privacy results

This report summarizes the exported results in [`privacy_results_seed-berka.csv`](./privacy_results_seed-berka.csv) and [`privacy_results_seed-berka_feature_engineered.csv`](./privacy_results_seed-berka_feature_engineered.csv). The calculations are per table; they exclude identifier columns and do not test disclosure through cross-table relationships.

## Executive summary

- The black-box DCR membership attack is near chance on most evaluated tables: the exported ROC-AUC values are mostly around 0.5. This is not evidence of privacy because the evaluation uses a random split of the seed data, not the actual synthesizer training records and a verified non-member holdout.
- Hitting Rate is very high for `account`, `card`, and `disp` in all runs. The `disp` table has only one evaluated categorical feature and two distinct values, so these results mainly reflect its extremely small domain, not the privacy of linked customer/account records.
- `trans` differs markedly between the regular and feature-engineered runs: Hitting Rate falls from about 0.65 to 0.00, Epsilon Identifiability Risk remains around 0.60, and median DCR falls from about 112 to 9.3. These are not directly comparable privacy improvements: the feature-engineered table has 25 evaluated features rather than 8, and its values and synthetic outputs come from a different data/training setup.
- The `results2` run was generated using `data/berka2`, but its values are in the Berka CSV and were computed against `data/berka`. Treat these as **mismatched-reference diagnostics**, not as a valid evaluation of Berka2. Berka2 needs a separate correctly configured evaluation before conclusions about that seed can be made.
- Median DCR is unavailable for `account` and `disp` in all three runs because the median real-to-real nearest-neighbor distance is zero. It is omitted rather than replaced with an artificial value.

## Data and evaluation caveats

The exports contain:

| Export | Synthetic run | Seed/reference used in the export | Important context |
|---|---|---|---|
| `privacy_results_seed-berka.csv` | `results/ClavaDDPM_notebook` | `berka` | The run is configured against the regular Berka data. |
| `privacy_results_seed-berka.csv` | `results2/ClavaDDPM_notebook` | `berka` | The synthesis notebook uses `data/berka2`, but this export evaluates it against `berka`. The Berka2 result is not correctly measured. |
| `privacy_results_seed-berka_feature_engineered.csv` | `results/feature_100k_run1/ClavaDDPM_experiment` | `berka_feature_engineered` | Its experiment metadata says the transaction table was sampled to 100,000 rows for training from 1,056,320 feature-engineered rows. |

The evaluation notebook defaults to a reproducible 80/20 random split of the selected real CSV, then caps each real partition at 5,000 rows and each synthetic table at 10,000 rows. The synthesizers were not trained using those exact evaluation partitions. In particular, the feature-engineered run's transaction training sample was 100,000 rows, while its privacy evaluation split the full 1,056,320-row feature-engineered transaction table. Therefore the following exported values are **exploratory only** and should not be used as membership-risk estimates:

- DCR train-versus-holdout score;
- NNDR `privacy_loss`;
- all membership-inference results (ROC-AUC, TPR, accuracy, and member/non-member distances).

Hitting Rate, Epsilon Identifiability Risk, median DCR, and the NNDR means still describe comparisons of the supplied rows under the notebook's preprocessing, but they are not formal privacy guarantees. The CSV's seed name describes the selected evaluation data, not proof of the actual training records used by each synthesizer.

## Results

All values are unitless except the reported distances, which are in the metric's preprocessed feature space. Values are rounded for display. In the membership column, `member/non-member (n)` gives mean DCR to the synthetic table for each class and the number of attack candidates per class.

### Regular Berka seed — `results/ClavaDDPM_notebook`

| Table | Hitting | EIR | DCR | Median DCR | NNDR mean (SE) | NNDR privacy loss (SE) | MIA AUC | TPR @ 10% FPR | MIA accuracy | MIA mean DCR, member/non-member (n/class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| account | 0.989 | 0.156 | 0.527 | — | 1.000 (1.16e-9) | 2.86e-7 (1.18e-8) | 0.507 | 0.000 | 0.507 | 1.414/1.414 (900) |
| card | 0.999 | 0.380 | 0.752 | 1.270 | 0.588 (0.0126) | 0.114312 (0.016711) | 0.513 | 0.000 | 0.475 | 0.001601/0.001584 (179) |
| client | 0.101 | 0.090 | 0.172 | 1.000 | 0.905 (0.0044) | -0.003417 (0.005214) | 0.504 | 0.000 | 0.500 | 1.254/1.264 (1074) |
| disp | 1.000 | 0.000 | 0.000 | — | 0.000 (0) | 0 (0) | 0.500 | 0.000 | 0.500 | 0/0 (1074) |
| district | 0.115 | 0.525 | 0.636 | 1.343 | 0.965 (0.0040) | 0.132811 (0.031381) | 0.527 | 0.188 | 0.625 | 2.034/2.056 (16) |
| loan | 0.354 | 0.501 | 0.787 | 28.505 | 0.999 (5.42e-5) | 0.001956 (0.000148) | 0.549 | 0.146 | 0.474 | 1.415/1.416 (137) |
| order | 0.568 | 0.504 | 0.796 | 65.917 | 1.000 (2.86e-7) | 2.59e-5 (1.45e-6) | 0.500 | 0.086 | 0.503 | 2.000/2.000 (1295) |
| trans | 0.650 | 0.595 | 0.563 | 112.063 | 1.000 (2.06e-6) | 5.97e-6 (2.96e-6) | 0.510 | 0.099 | 0.510 | 2.829/2.829 (4000) |

### `results2/ClavaDDPM_notebook` — exported against Berka, synthesis input was Berka2

This table preserves the CSV values so the run is not hidden, but it is **not a valid Berka2 evaluation**. The generating notebook points at `data/berka2`, while the comparison/export labels this run as `berka` and uses the `berka` reference data.

| Table | Hitting | EIR | DCR | Median DCR | NNDR mean (SE) | NNDR privacy loss (SE) | MIA AUC | TPR @ 10% FPR | MIA accuracy | MIA mean DCR, member/non-member (n/class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| account | 0.998 | 0.171 | 0.528 | — | 1.000 (8.32e-10) | 2.84e-7 (1.04e-8) | 0.504 | 0.000 | 0.507 | 1.414/1.414 (900) |
| card | 0.997 | 0.418 | 0.751 | 1.319 | 0.574 (0.0118) | 0.092230 (0.015620) | 0.499 | 0.000 | 0.492 | 0.001251/0.001229 (179) |
| client | 0.105 | 0.096 | 0.170 | 1.000 | 0.907 (0.0041) | -0.009919 (0.004901) | 0.504 | 0.000 | 0.500 | 1.260/1.272 (1074) |
| disp | 1.000 | 0.000 | 0.000 | — | 0.000 (0) | 0 (0) | 0.500 | 0.000 | 0.500 | 0/0 (1074) |
| district | 0.131 | 0.590 | 0.597 | 1.349 | 0.960 (0.0044) | 0.153849 (0.033426) | 0.465 | 0.062 | 0.562 | 2.035/2.055 (16) |
| loan | 0.422 | 0.565 | 0.795 | 28.505 | 0.999 (4.58e-5) | 0.002064 (0.000142) | 0.504 | 0.109 | 0.453 | 1.415/1.415 (137) |
| order | 0.605 | 0.545 | 0.796 | 65.917 | 1.000 (2.86e-7) | 2.59e-5 (1.45e-6) | 0.495 | 0.071 | 0.496 | 2.000/2.000 (1295) |
| trans | 0.648 | 0.584 | 0.556 | 112.063 | 1.000 (2.01e-6) | 7.47e-6 (3.13e-6) | 0.507 | 0.087 | 0.511 | 2.829/2.829 (4000) |

### Feature-engineered Berka seed — `results/feature_100k_run1/ClavaDDPM_experiment`

| Table | Hitting | EIR | DCR | Median DCR | NNDR mean (SE) | NNDR privacy loss (SE) | MIA AUC | TPR @ 10% FPR | MIA accuracy | MIA mean DCR, member/non-member (n/class) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| account | 0.998 | 0.168 | 0.515 | — | 1.000 (1.01e-9) | 2.84e-7 (1.00e-8) | 0.494 | 0.000 | 0.500 | 1.414/1.414 (900) |
| card | 0.999 | 0.407 | 0.719 | 1.283 | 0.581 (0.0125) | 0.120999 (0.016422) | 0.534 | 0.000 | 0.486 | 0.001264/0.001430 (179) |
| client | 0.107 | 0.099 | 0.173 | 1.000 | 0.903 (0.0043) | -0.007113 (0.005082) | 0.500 | 0.000 | 0.500 | 1.260/1.260 (1074) |
| disp | 1.000 | 0.000 | 0.000 | — | 0.000 (0) | 0 (0) | 0.500 | 0.000 | 0.500 | 0/0 (1074) |
| district | 0.148 | 0.656 | 0.649 | 1.340 | 0.966 (0.0040) | 0.106090 (0.028980) | 0.508 | 0.125 | 0.500 | 2.028/2.047 (16) |
| loan | 0.413 | 0.569 | 0.804 | 28.505 | 0.999 (5.14e-5) | 0.002171 (0.000121) | 0.525 | 0.161 | 0.482 | 1.415/1.415 (137) |
| order | 0.629 | 0.565 | 0.791 | 65.917 | 1.000 (3.38e-7) | 2.75e-5 (1.64e-6) | 0.483 | 0.080 | 0.476 | 2.000/2.000 (1295) |
| trans | 0.000 | 0.601 | 0.488 | 9.294 | 0.997 (6.37e-5) | 0.005407 (0.000488) | 0.489 | 0.097 | 0.478 | 2.846/2.846 (5000) |

## Discussion

### Black-box membership inference

Across all tables and exports, ROC-AUC ranges from approximately 0.465 to 0.549, close to the 0.5 random-ranking baseline. Most calibrated accuracies are near 0.5 as well. The district results use only 16 candidates per class, so their accuracy and TPR estimates are particularly unstable. These attack results say only that this specific distance-based attack did not reliably rank the random-split labels; they do not establish that training membership is hidden. Re-run using the actual membership partition before drawing a privacy conclusion.

### Hitting Rate and Epsilon Identifiability Risk

Hitting Rate is near 1 for `account`, `card`, and `disp`, meaning most reference rows have at least one synthetic row within the configured tolerance. This is a reason to inspect those tables, but not proof of memorization: the measures omit IDs, and `disp` has just one feature with two values. Its Hitting Rate of 1 coexists with EIR 0 because a common two-value categorical distribution provides little individual-level distinguishability under these metrics.

The largest EIR values are in `district`, `loan`, `order`, and `trans` (up to 0.601 for feature-engineered transactions). EIR is a nearest-neighbor competition score, not a calibrated probability that a person can be identified. The feature-engineered transaction Hitting Rate of 0 and EIR of 0.601 are not contradictory: Hitting requires a near-match across all 25 features, while EIR compares relative nearest-neighbor distances.

### Distance and neighborhood metrics

Median DCR is a ratio to real-data nearest-neighbor spacing, not a percentage. Values such as 112 for regular Berka transactions and 9.29 for feature-engineered transactions are strongly affected by dimensions, distributions, and preprocessing; they are not a direct safety ranking across different schemas. Feature-engineered transactions have 25 features versus 8 in regular Berka, and come from a larger, differently sampled input. Median DCR is undefined for `account` and `disp` because the denominator is zero.

NNDR means near 1 for several tables indicate that the nearest and second-nearest real records are similarly distant for synthetic rows; a small mean alone does not establish privacy. The NNDR privacy-loss column is intended to compare training versus true holdout neighborhood ratios. Its positive value for feature-engineered transactions (0.005407, standard error 0.000488) would warrant follow-up with a valid holdout; under the present random-split protocol, it cannot be attributed to training-set leakage.

### DCR train/holdout score

DCR is a fraction of synthetic rows whose nearest training row is closer than their nearest holdout row. Its reference value depends on the sizes of the two comparison sets, not always 0.5. The notebook caps those sets, and in this report the split is not the actual training/holdout split used by synthesis. Thus the exported DCR scores (for example 0.488 for engineered transactions or 0.527 for regular Berka accounts) should not be read as privacy-safe/unsafe thresholds.

### Berka versus feature-engineered results

The regular Berka `trans` row has 8 domain features and reports Hitting 0.650, EIR 0.595, and median DCR 112.063. The feature-engineered row has 25 domain features and reports Hitting 0.000, EIR 0.601, and median DCR 9.294. The changed schema and different training setup mean these differences combine feature engineering, changed source data/sample size, and generator behavior. They do not isolate the effect of feature engineering on privacy. A controlled comparison requires holding the underlying train/holdout records, generator configuration, and sample sizes fixed, changing only the feature representation, and evaluating each run against its actual training membership.

## Appendix: metric definitions and calculations

The calculations follow the project’s [`multi_table_privacy_comparison.ipynb`](./multi_table_privacy_comparison.ipynb), which reuses the implementations in the [privacy evaluation pipeline](../../evaluation/privacy/privacy_evaluation_pipeline.ipynb) and [membership-inference notebook](../../evaluation/privacy/membership_inference_attack.ipynb). The per-table columns and continuous/discrete types come from each table's `*_domain.json`; IDs are therefore excluded. For the SynthEval metrics, categorical values are ordinal encoded and numerical values min-max encoded. For distance metrics and the black-box attack, categoricals are one-hot encoded and numeric features are range-scaled.

Let \(R\) denote real reference rows, \(S\) synthetic rows, \(R_{train}\) training-reference rows, and \(R_{holdout}\) holdout rows. In this evaluation, `R_train` and `R_holdout` are random partitions/capped samples, not verified model-membership partitions.

### Hitting Rate

For each real row, a hit occurs if at least one synthetic row matches every categorical feature exactly and is within 3% of the real-data range for every continuous feature. The score is the fraction of real rows hit (`hitting_rate`). Lower is preferable as a replication indicator; it is not itself a privacy guarantee.

### Epsilon Identifiability Risk (EIR)

For each real row, compare its nearest synthetic neighbor with its nearest *other* real neighbor using the Gower distance. EIR is the fraction of real rows whose synthetic neighbor is closer (`epsilon_identifiability_risk`). The implementation weights rare values by inverse column entropy. Lower values are generally preferable; this is a relative-distance risk indicator.

### Distance to Closest Record (DCR)

For each synthetic row \(s\), compute its nearest distance to \(R_{train}\) and \(R_{holdout}\). DCR is the fraction of synthetic rows for which the training distance is smaller:

\[
\operatorname{DCR} =
\frac{1}{|S|}\sum_{s\in S}
\mathbf{1}\left[d(s,R_{train}) < d(s,R_{holdout})\right].
\]

This is the exported `dcr_score`. The toolkit's no-overfitting reference is approximately
\(|R_{train}|/(|R_{train}|+|R_{holdout}|)\), so it changes with partition sizes. A valid interpretation requires the actual training records and a genuinely unused holdout.

### Median DCR

Compute each synthetic row's nearest distance to the real training rows, and each real training row's nearest distance to a *different* real training row. Divide the median of the first set by the median of the second. This is the exported `median_dcr_score`. A score below 1 means synthetic rows are closer to real rows than typical real-to-real spacing; above 1 means farther away. The value depends on feature space and preprocessing. If the denominator median is zero, the score is undefined and is omitted.

### Nearest Neighbor Distance Ratio (NNDR)

For each synthetic row, find its nearest and second-nearest real rows and compute \(d_1/d_2\). The reported `mean_nndr` is accompanied by `nndr_standard_error`. When a holdout is supplied, the toolkit reports
\[
\operatorname{privacy\_loss}
= \operatorname{meanNNDR}(R_{train})
- \operatorname{meanNNDR}(R_{holdout}),
\]
and a pooled `privacy_loss_standard_error`. A more positive difference can indicate synthetic points are more specifically close to the training set, but the result requires a true holdout and should be interpreted with the NNDR definition and uncertainty.

### Black-box DCR membership-inference attack

The attack samples a balanced set of real rows labelled member/non-member, computes each candidate's minimum L2 distance to the synthetic table, and uses negative distance as the membership score (closer means more likely member). The output includes:

- **ROC-AUC (`roc_auc`):** ranking ability; 0.5 is chance.
- **TPR at 10% FPR (`tpr_at_10_fpr`):** true-positive rate at a threshold with false-positive rate at most 10%.
- **Calibrated test accuracy (`calibrated_test_accuracy`):** Youden's \(J = \mathrm{TPR}-\mathrm{FPR}\) selects a threshold on a calibration half, then accuracy is measured on a separate test half.
- **Mean member/non-member DCR and candidates per class** (`mean_member_dcr`, `mean_non_member_dcr`, `candidates_per_class`): diagnostic distances and balanced sample size.

This is one simple attack, not a comprehensive privacy audit. It requires member examples actually used by the generator and non-member examples that were not used.

## Reproducibility and next steps

1. Re-evaluate the `results2` run with a distinct `berka2` seed selection and the `berka2` real data/domain metadata. Do not treat its current Berka-export values as Berka2 metrics.
2. Provide the exact training records/partition used by each run and a disjoint holdout, especially the 100,000-row transaction sample for the feature-engineered experiment.
3. Keep the same train/holdout sizes and sampled rows across runs, and record those counts and the chosen seed alongside each export.
4. For feature-engineering comparisons, fix the underlying records and model settings and change only the feature set. Compare per-table values within the same schema; avoid ranking scores across different feature dimensions.
5. Treat these measures as empirical diagnostics. None provides differential privacy or a formal guarantee against re-identification.
