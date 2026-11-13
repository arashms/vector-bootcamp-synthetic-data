# Multi-table synthetic banking data (ClavaDDPM): tuning summary

Date: 2026-11-13. Status: experiments 1–12 done; hyperparameter tuning complete and the
final configuration confirmed on 5 training seeds. Next: structural work on overdrafts.

## TL;DR

- We generate synthetic bank data and test whether an overdraft detector trained on it
  works on real data.
- Two fixes made the synthetic data structurally and chronologically valid:
  - a bug fix in how linked tables are joined;
  - a new way of encoding dates.
- Of all the hyperparameters tested (clustering, guidance, diffusion and classifier
  settings), only longer diffusion training and a larger diffusion batch gave consistent
  improvements: detector ROC-AUC 0.68 → **0.757 ± 0.042** (5 training seeds), against
  0.87 when trained on real data. Classifier settings made no difference.
- The remaining gap is mainly structural. Overdrafts are spread over 30–40% of synthetic
  accounts instead of 5.6%, because the model does not generate an account's
  transactions as a sequence. Tuning alone is unlikely to close this.

## 1. Goal and setup

- **Goal:** generate a synthetic copy of a multi-table bank dataset (Berka: 8 linked tables)
  that is realistic enough to **train an overdraft detector that works on real data**.
- **Data:** a cohort of 500 accounts with their full histories (117,560 transactions;
  28 accounts, 5.6%, ever go overdrawn).
- **Model:** ClavaDDPM, which uses clustering plus per-table diffusion models and
  guidance classifiers.
- **Primary metric (transfer):** a detector is trained on synthetic transactions and
  scored on real held-out transactions (ROC-AUC, AP standardized to real prevalence).
  For reference, a detector trained on real data scores **ROC-AUC 0.872, AP 0.242**.
- **Secondary metrics (fidelity):** per-table KS/TVD, cross-table correlation,
  cardinality, key validity, and domain checks (accounts ever overdrawn, transactions
  dated before account opening).
- **Method:**
  - One change at a time, compared with the current baseline.
  - Each comparison uses **3 training seeds (42/43/44), paired by seed**.
  - A change is adopted only if it improves the primary metric consistently
    (at least 2 of 3 pairs) without breaking structure.

### How to read the tables

- Cells like `0.762 / 0.680 / 0.700` are the values for training seeds 42 / 43 / 44.
  Each column pair compares the same seed with and without the change.
- ↑ means higher is better, and ↓ means lower is better. **Bold** marks the better value.
- For reference, a random detector scores ROC-AUC 0.5 and standardized AP ≈ 0.0017,
  which is the real overdraft rate in the test set.

### Glossary

| Term | Meaning |
|---|---|
| ClavaDDPM | Multi-table synthetic data generator. It clusters child rows by parent, then trains one diffusion model per table plus classifiers that steer generation ("guidance"). |
| Diffusion model / iterations | The neural network that generates rows of one table. Iterations are training steps. |
| Guidance strength | How strongly the classifiers push generated child rows toward their parent's cluster. |
| Clusters / parent weight | ClavaDDPM groups child rows into latent clusters to link tables. Parent weight sets how much parent columns influence those clusters. |
| Training seed / sampling seed | A training seed retrains all models; a sampling seed only regenerates data from the same models. |
| Disposition (`disp`) | Berka table linking a client to an account (owner or user). Cards hang off dispositions. |
| Multi-parent matching | `disp` has two parents (client and account), which are generated separately and must be joined afterwards. |
| Orphan card | A card whose disposition no longer exists, which makes it a broken link. |
| FK validity | Share of foreign keys that point to an existing parent row (1.0 = all valid). |
| KS (Kolmogorov–Smirnov) | Distance between real and synthetic distributions of a numeric column (0 = identical, 1 = disjoint). |
| TVD (total variation distance) | The same idea for categorical columns (0 = identical). |
| Correlation RMSE | Error between real and synthetic correlations of parent and child columns (0 = identical). |
| Cardinality KS | KS on the number of children per parent, e.g. transactions per account. |
| ROC-AUC | How well the detector ranks overdraft transactions above normal ones (0.5 = random, 1 = perfect). |
| Standardized AP | Average precision, adjusted to the real test set's overdraft rate so runs are comparable. Random ≈ 0.0017. |

## 2. Key finding about noise

Retraining the same configuration with a different seed moves the detector ROC-AUC by
**0.61–0.77 (sd ≈ 0.08)**. By contrast, changing the sampling seed moves it by only ±0.007.
Early sweeps that used one training seed therefore could not tell settings apart.
**All decisions since then use 3 paired training seeds.**

## 3. Timeline of experiments and decisions

| # | Change tested | Result | Decision |
|---|---|---|---|
| 1 | Initial sweeps: guidance strength (0–4), cluster count (10–50), parent weight (0.5–2) | All within noise; guidance 2 and 4 degrade fidelity and ROC-AUC | Keep defaults (guidance 1, 50 clusters, weight 1) |
| 2 | **Fix multi-parent matching** (toolkit bug) | Orphan cards 193 → 0; key validity 0.85 → 1.0 | **Adopted** |
| 3 | Dates stored as days since account opening | Pre-opening transactions → 0%, but ~22% fall after the data end | Rejected |
| 4 | **Dates as a fraction of the account's window** | Pre-opening and post-end transactions both 0%; cross-table date correlation 4× better | **Adopted** |
| 5 | Extra account feature `ever_overdrawn` | Learned at the account level but not passed on to transactions | Rejected |
| 6 | Recompute balance as a running sum | Overdrawn accounts 40% → 80% | Rejected (CPU check only) |
| 7 | Clustering `kmeans` / `gmm` vs `kmeans_and_gmm` | `kmeans` better marginals but lower ROC-AUC in 2/3; `gmm` within noise | Keep `kmeans_and_gmm` |
| 8 | **Diffusion iterations 2500 / 5000 / 10000** | 10000 better in 3/3 seeds on ROC-AUC, AP, fidelity | **Adopted 10000** |
| 9 | Diffusion iterations 20000 | Better fidelity; ROC-AUC lower in 2/3 | Keep 10000 |
| 10 | **Diffusion learning rate, size, dropout, timesteps, batch size** | Batch 512 better in 3/3 seeds on ROC-AUC and AP; dropout, low LR and the smaller model hurt; others within noise | **Adopted batch 512** |
| 11 | Classifier learning rate, iterations, size, batch, time embedding | No setting improves ROC-AUC and AP in 3/3 seeds; all within noise; none helps overdraft concentration | Keep defaults |
| 12 | Final configuration on 5 training seeds | ROC-AUC 0.757 ± 0.042, AP 0.0051 ± 0.0016; new seeds match | Confirmed |

- Experiment 1 used one training seed and three sampling seeds, before the noise
  finding above, so its "no effect" conclusions are weak.
- Experiment numbers differ from the internal step labels in PROGRESS.md and the
  notebooks: experiments 2–4 = steps 1/3/3b, 5–6 = step 4 and the running-balance check,
  7–12 = steps 5–10.

## 4. Adopted changes in detail (seed 42 / 43 / 44)

### 4.1 Matching fix (structural correctness)

Clients and accounts are generated separately, and each produces its own copy of
the `disp` link table. The toolkit kept whichever copy was smaller. Cards had been
generated from the client-side copy, so they could end up pointing to dispositions
that no longer existed (orphans) or to the wrong ones. The fix keeps the client-side
copy and assigns each disposition to one account slot without reuse.

| Metric | Before | After |
|---|---|---|
| Orphan cards (cards pointing to missing dispositions) | 0 / 11 / 0 | 0 / 0 / 0 |
| Minimum foreign-key validity | 1.0 / 0.851 / 1.0 | 1.0 / 1.0 / 1.0 |
| Accounts without a disposition | 107 / 0 / 44 | 60 / 0 / 0 |

On 43 saved runs replayed on CPU, orphan cards fell from 193 to 0.

### 4.2 Window-fraction dates (chronology)

Originally each date was generated as an absolute calendar date, independently of the
account's opening date. As a result, about 20% of transactions happened before their
account existed. Now each child date is stored as a number from 0 to 1: 0 is the
account's opening date and 1 is the end of the dataset. Every generated date therefore
falls inside the account's lifetime by construction.

| Metric | Real | Before | After |
|---|---|---|---|
| Transactions before account opening | 0% | 20.5 / 23.1 / 16.1% | **0 / 0 / 0%** |
| Account→transaction correlation RMSE ↓ | – | 0.110 / 0.134 / 0.093 | **0.029 / 0.027 / 0.019** |
| Loan vs account-opening date correlation | 0.958 | −0.12 / 0.04 / −0.13 | **0.90 / 0.84 / 0.82** |
| Calendar transaction-date KS ↓ | – | 0.005 / 0.014 / 0.044 | 0.129 / 0.129 / 0.109 |

The cost is a worse calendar-date marginal. In exchange, chronology is valid by construction.

### 4.3 Diffusion iterations 5000 → 10000

| Metric | Real | 5000 | 10000 |
|---|---|---|---|
| Detector ROC-AUC ↑ | 0.872 | 0.661 / 0.659 / 0.658 | **0.762 / 0.680 / 0.700** |
| Standardized AP ↑ | 0.242 | 0.0033 / 0.0030 / 0.0037 | **0.0040 / 0.0033 / 0.0039** |
| Transaction TVD ↓ | – | 0.023 / 0.023 / 0.031 | **0.018 / 0.014 / 0.018** |
| All-table numerical KS ↓ | – | 0.097 / 0.109 / 0.131 | **0.076 / 0.096 / 0.093** |
| Accounts ever overdrawn | 5.6% | 47 / 48 / 40% | **41 / 31 / 33%** |

### 4.4 Diffusion iterations 10000 → 20000 (not adopted)

| Metric | 10000 | 20000 |
|---|---|---|
| Detector ROC-AUC ↑ | 0.762 / 0.680 / 0.700 | 0.691 / 0.700 / 0.680 |
| Standardized AP ↑ | 0.0040 / 0.0033 / 0.0039 | 0.0040 / 0.0045 / 0.0040 |
| All-table categorical TVD ↓ | 0.096 / 0.084 / 0.100 | 0.091 / 0.078 / 0.076 |
| Accounts ever overdrawn (real 5.6%) | 41 / 31 / 33% | 25 / 28 / 24% |

20000 is the best setting for fidelity. It is not adopted because transfer, the goal,
did not improve, and training time doubles.

### 4.5 Diffusion hyperparameter sweep (experiment 10)

One change at a time against the 10000-iteration baseline (learning rate 0.0006,
batch 256, dropout 0, 500 timesteps). "(n/3)" counts the seed pairs better than baseline.

| Setting | Detector ROC-AUC ↑ | Standardized AP ↑ | Transaction TVD ↓ | Accounts ever overdrawn (real 5.6%) |
|---|---|---|---|---|
| baseline | 0.762 / 0.680 / 0.700 | 0.0040 / 0.0033 / 0.0039 | 0.018 / 0.014 / 0.018 | 41 / 31 / 33% |
| **batch 512** | **0.770 / 0.691 / 0.806 (3/3)** | **0.0051 / 0.0036 / 0.0074 (3/3)** | **0.012 / 0.012 / 0.013 (3/3)** | 38 / 29 / 42% (2/3) |
| batch 128 | 0.797 / 0.661 / 0.686 (1/3) | 0.0077 / 0.0030 / 0.0040 (2/3) | 0.026 / 0.020 / 0.023 (0/3) | 65 / 39 / 39% (0/3) |
| smaller model | 0.731 / 0.735 / 0.709 (2/3) | 0.0037 / 0.0035 / 0.0034 (1/3) | 0.021 / 0.021 / 0.020 (0/3) | 62 / 66 / 57% (0/3) |
| dropout 0.1 | 0.647 / 0.648 / 0.586 (0/3) | 0.0029 / 0.0029 / 0.0032 (0/3) | 0.023 / 0.024 / 0.020 (0/3) | 62 / 54 / 41% (0/3) |
| dropout 0.2 | 0.724 / 0.708 / 0.695 (1/3) | 0.0038 / 0.0032 / 0.0028 (0/3) | 0.038 / 0.036 / 0.033 (0/3) | 91 / 76 / 93% (0/3) |
| learning rate 0.0002 | 0.666 / 0.547 / 0.585 (0/3) | 0.0032 / 0.0026 / 0.0022 (0/3) | 0.021 / 0.020 / 0.022 (0/3) | 55 / 32 / 70% (0/3) |
| learning rate 0.001 | 0.777 / 0.671 / 0.712 (2/3) | 0.0042 / 0.0034 / 0.0040 (3/3) | 0.018 / 0.015 / 0.017 (2/3) | 68 / 34 / 34% (0/3) |
| 250 timesteps | 0.708 / 0.720 / 0.693 (1/3) | 0.0041 / 0.0039 / 0.0042 (3/3) | 0.019 / 0.018 / 0.016 (1/3) | 32 / 34 / 36% (1/3) |
| 1000 timesteps | 0.669 / 0.688 / 0.740 (2/3) | 0.0038 / 0.0078 / 0.0045 (2/3) | 0.017 / 0.015 / 0.018 (1/3) | 35 / 36 / 43% (1/3) |

- Batch 512 is the only setting that improves both transfer metrics in every seed:
  mean ROC-AUC 0.714 → 0.755 and standardized AP 0.0037 → 0.0054.
- It also improves most fidelity metrics in 2–3 of 3 seeds.
- Regularization (dropout), a lower learning rate and a smaller network all make
  the data worse. They spread overdrafts over up to 93% of accounts.

### 4.6 Classifier sweep (experiment 11, not adopted)

One change at a time against the batch-512 baseline (classifier learning rate 1e-4,
5000 iterations, batch 256, time embedding 128). The baseline reproduced exactly.

| Setting | Detector ROC-AUC ↑ | Standardized AP ↑ |
|---|---|---|
| baseline | 0.770 / 0.691 / 0.806 | 0.0051 / 0.0036 / 0.0074 |
| learning rate 3e-5 | 0.770 / 0.702 / 0.832 (2/3) | 0.0045 / 0.0055 / 0.0077 (2/3) |
| learning rate 3e-4 | 0.722 / 0.675 / 0.807 (1/3) | 0.0040 / 0.0031 / 0.0075 (1/3) |
| 2500 iterations | 0.754 / 0.667 / 0.825 (1/3) | 0.0042 / 0.0033 / 0.0069 (0/3) |
| 10000 iterations | 0.751 / 0.664 / 0.811 (1/3) | 0.0047 / 0.0034 / 0.0076 (1/3) |
| smaller network | 0.798 / 0.692 / 0.825 (3/3) | 0.0052 / 0.0037 / 0.0073 (2/3) |
| batch 128 | 0.755 / 0.697 / 0.823 (2/3) | 0.0043 / 0.0058 / 0.0074 (1/3) |
| batch 512 | 0.776 / 0.679 / 0.825 (2/3) | 0.0048 / 0.0040 / 0.0076 (2/3) |
| time embedding 64 | 0.759 / 0.694 / 0.818 (2/3) | 0.0045 / 0.0050 / 0.0077 (2/3) |
| time embedding 256 | 0.792 / 0.650 / 0.827 (2/3) | 0.0049 / 0.0032 / 0.0078 (1/3) |

- All differences are within training noise. The smaller network raises ROC-AUC in
  3/3 seeds (+0.016 mean) but not AP, and worsens transaction fidelity in 3/3.
- No setting reduces overdraft spread. The classifier is not the bottleneck.

### 4.7 Final configuration on 5 training seeds (experiment 12)

| Seed | 42 | 43 | 44 | 45 | 46 | Mean ± sd |
|---|---|---|---|---|---|---|
| Detector ROC-AUC ↑ (real 0.872) | 0.770 | 0.691 | 0.806 | 0.768 | 0.751 | **0.757 ± 0.042** |
| Standardized AP ↑ (real 0.242) | 0.0051 | 0.0036 | 0.0074 | 0.0055 | 0.0037 | **0.0051 ± 0.0016** |
| Transaction TVD ↓ | 0.012 | 0.012 | 0.013 | 0.012 | 0.016 | 0.013 ± 0.002 |
| All-table numerical KS ↓ | 0.076 | 0.089 | 0.096 | 0.087 | 0.089 | 0.087 ± 0.008 |
| Accounts ever overdrawn (real 5.6%) | 38.3% | 29.3% | 41.8% | 38.9% | 29.4% | 35.5 ± 5.8% |

The two new seeds (45, 46) fall inside the range of the first three, so the earlier
conclusions hold. Structure is valid in every seed (0% before opening, FK validity 1.0).

## 5. Where we stand

Final configuration confirmed on 5 training seeds (42–46, report `c480bd90c664b936`):
ROC-AUC **0.757 ± 0.042**, standardized AP **0.0051 ± 0.0016**, accounts ever
overdrawn 35.5 ± 5.8%, before-opening 0% and FK validity 1.0 in every seed.
Seed-to-seed ROC-AUC spread halved versus the first baseline (sd 0.081 → 0.042).

| Metric | Real | First baseline (3 seeds) | Final (5 seeds) |
|---|---|---|---|
| Detector ROC-AUC (mean) ↑ | 0.872 (random 0.5) | 0.679 (sd 0.081) | **0.757 (sd 0.042)** |
| Standardized AP ↑ | 0.242 (random ≈ 0.0017) | 0.0042 | **0.0051** |
| Transactions before account opening | 0% | ~20% | **0%** |
| Orphan cards / FK validity | 0 / 1.0 | up to 11 / 0.85 | **0 / 1.0** |
| Accounts ever overdrawn | 5.6% | 44–58% | 29–42% (mean 35.5%) |

- The ROC-AUC gain is cumulative. The date encoding alone lowered it slightly
  (0.679 → 0.659). 10000 iterations raised it to 0.714, and batch 512 to 0.755
  (0.757 over 5 seeds).
- Detectors trained on synthetic data are better than random but far below real-trained ones.

Current configuration: fixed multi-parent matching, `fraction_of_window` dates,
`kmeans_and_gmm` clustering with 50 clusters, guidance 1, 10000 diffusion iterations
and diffusion batch size 512.

## 6. Conclusions

1. **The biggest gains came from fixing data representation and bugs, not from
   hyperparameter tuning.** Matching and date encoding fixed structural and
   chronological errors completely.
2. **Only two hyperparameters gave consistent transfer gains:** diffusion training length
   (5000 → 10000) and diffusion batch size (256 → 512). Clustering, guidance, parent
   weight, learning rate, timesteps and all classifier settings are within training noise. Dropout and smaller
   networks hurt.
3. **The transfer gap is still large.** Standardized AP is ~0.005 vs 0.242 on real data
   (~47×). ROC-AUC is 0.76 vs 0.87.
4. **The main remaining failure is overdraft concentration.** Real overdrafts occur in
   5.6% of accounts. Synthetic overdrafts are spread thinly across about 30–40% of accounts.
   ClavaDDPM generates transactions largely independently within an account, so it
   does not model balances as a sequence. Tuning reduces this gap but is unlikely
   to close it.
5. **Training-seed variance is large** (ROC-AUC sd 0.08 at the first baseline, 0.04 for
   the final configuration). Any claimed improvement needs several paired seeds.
   Single-run comparisons are unreliable.

## 7. Next steps

1. Hyperparameter tuning is complete (experiments 1–12). Only optional leftover:
   diffusion batch 1024, since 512 was the edge of the tested range.
2. The remaining gap is structural (overdraft concentration). Planned: a CPU diagnosis of
   how synthetic overdrafts are distributed, then one of:
   - an account-level balance level generated by the account model, with transaction
     balances stored relative to it;
   - conditioning transactions explicitly on an account-level overdraft profile;
   - a sequence-aware transaction model (largest effort, outside ClavaDDPM).

## Appendix: compute

- Node: 8× H200. GPUs 1–7 are used; GPU 0 is shared with another user.
- One baseline run (10000 iterations) takes about 10 minutes end to end.
- Runs are cached by stage, and reruns are bit-identical.
- Several jobs share each GPU (each uses < 1 GB and ~25% of the GPU), which roughly
  triples throughput.
