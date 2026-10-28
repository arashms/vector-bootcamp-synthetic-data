# Multi-table hyperparameter tuning

Start with [guidance_strength.ipynb](guidance_strength.ipynb). It reuses the trained ClavaDDPM models to explore guidance strength without retraining.

Hyperparameters are settings we choose to control how the model groups, learns, or generates data. The goal is to improve relationships between tables while keeping individual columns realistic and preserving data diversity.

## Start here: transaction and overdraft tuning

1. Open [transaction_overdraft_evaluation.ipynb](transaction_overdraft_evaluation.ipynb)
   to inspect the five **already generated** guidance runs. This uses CPU and takes
   about 1–3 minutes. No loan classification is used.
2. Open [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb) to preview a new
   guidance, clustering, diffusion, or classifier sweep. `RUN_NEW_TRIALS = False`
   keeps GPU work disabled until a budget is chosen.
3. Read [PROGRESS.md](PROGRESS.md) for actual findings, decisions, and validation
   limits. Original training/synthesis notebooks, YAML configs, and baseline
   checkpoints are preserved.

Use the project `.venv` kernel. Restore the existing declared dependencies with
`uv sync --locked --group tabular-data --inexact` if needed.

### Minimal code structure

| File | Responsibility |
| --- | --- |
| `transaction_overdraft_evaluation.ipynb` | Saved-run comparison, plots, diagnostics, and conclusion. |
| `hyperparameter_tuning.ipynb` + `tuning.py` | Choose one experiment, preview reuse, explicitly execute selected trials. |
| `../evaluation/overdraft_transfer_evaluator.py` | Existing real-trained random forest; no target or ID inputs. |
| `../evaluation/transaction_quality.py` + `transaction_evaluation.py` | Shared fidelity checks, bootstrap uncertainty, report saving. |
| `guidance_strength.ipynb` | Historical all-table guidance experiment and its original metrics. |

### What the current evaluation means

The detector learns from real transactions, then runs unchanged on held-out real
accounts and each synthetic run. Its target is **current negative balance**;
this is not future-overdraft forecasting. AP, ROC-AUC, fixed-threshold precision/
recall, counts, and prevalence are reported. Standardized AP reweights every run
to the real test event rate, removing only the class-prior difference.

The quality checks compare date, amount, balance, and transaction categories;
repeat comparisons inside the overdraft class; inspect category-specific and
monthly overdraft rates; and check account volumes, transaction counts including
zeros, account/transaction relationships, key integrity, and date chronology.
Numerical and categorical distances exclude identifiers. Constant/undefined
correlation pairs remain visible rather than being scored as perfect matches.

There is **no combined score or automatic winner**. Compare rare-event behavior,
overall fidelity, and account consistency separately. Account-bootstrap intervals
condition on the fitted detector and saved generation; repeat promising settings
across generation seeds and eventually training seeds.

### Checkpoint reuse and reproducibility

| Changed setting | Work repeated |
| --- | --- |
| Guidance or sampling seed | Generation only; reuse all models. |
| Classifier settings | Classifier training and generation; reuse diffusion and clustering. |
| Diffusion settings | Diffusion, classifier, and generation; reuse clustering. |
| Clustering or training seed | Clustering and all downstream stages. |

The helper uses toolkit functions for each stage. Fresh stages use separate,
per-relation random seeds, so cache reuse does not alter subsequent random draws.
The toolkit still fixes some internal splits/random states; a training-seed change
is not a different real train/test partition. Cross-hardware determinism is not
guaranteed.

Fresh training aligns diffusion/classifier internal splits to **70/20/10** because
the toolkit fits their preprocessing independently. Original diffusion used
99/0.5/0.5 while its classifier defaulted to 70/20/10. Establish a separate fresh
baseline before attributing improvements to another setting. Legacy checkpoint
mode accepts only sampling/matching changes and verifies the current input tables
against the saved clustering checkpoint.

Artifacts live under `../results/experiments/transaction_tuning/`, with separate
`clustering/`, `diffusion/`, `models/`, and `samples/` directories. Their IDs include
input data, relevant settings, seeds, source fingerprints, and dependency versions.
`manifest.json` records inputs; `status.json` records completion/failure, duration,
and artifact checksums. Complete matching stages are reused; changed artifacts
are rejected. Failed stages require a fresh experiment root to retry; there is no
optimizer-state resume. Use one executing notebook process at a time.

Actual matched tables are captured before toolkit cleanup removes keys. Reports
live separately under `../results/experiments/transaction_overdraft/<fingerprint>/`
and contain a comparison CSV, real account split, predictions, per-run detail
CSVs, metadata, and plots. Pickles are for trusted local experiment artifacts.

### Research limits and next decision

Saved-data evaluation has executed on CPU; full fresh generator training has
**not** run under the current budget. The real test has only eight overdrafts,
so its AP estimate has a wide interval. Existing generators saw the full
reference data; a future synthetic-trained → real-test utility study needs an
account holdout excluded **before** generator training and clustering.

Sampled transactions are incomplete histories. Rolling/history features,
sequential balance checks, and graph metrics are deferred until their meaning
for this sample is established. This framework does not measure privacy.

For the current saved runs, 0–0.5 guidance is promising for overall transaction
fidelity, but no overdraft winner is established. All five runs have serious
account-date chronology errors. See the evaluation notebook conclusion.

Run the targeted regression checks from the repository root:

```bash
.venv/bin/python -m pytest -q implementations/tabular_data/multi_table/tests/test_transaction_workflow.py
```

## Historical guidance notebook

The earlier notebook sweeps guidance strength and evaluates all eight parent–child
relationships plus individual column distributions. Its scores and artifacts are
retained as context; the transaction-focused notebook is now the selection view.

## Run the guidance experiment

1. Open the notebook using the project's `.venv` kernel with GPU access. Existing checkpoints must be present in `../results/models/` and `../results/cluster_ckpt.pkl`.
2. Adjust the settings cell:

   ```python
   GUIDANCE_SCALES = [0.0, 0.5, 1.0, 2.0, 4.0]
   SEEDS = [42]
   SWEEP_NAME = "guidance_strength_first_pass"
   ```

3. Run the cells in order. Five guidance values and one seed produce five full-dataset trials. Keep the other settings fixed so the comparison measures guidance strength.
4. Review the comparison table and inspect individual edge or column scores. Repeat promising values with more seeds, such as `[42, 123, 456]`, before choosing a setting.

Each trial saves under:

```text
../results/experiments/<SWEEP_NAME>/guidance_<value>_seed<seed>/
```

Completed trials are reused when their saved settings and checkpoint signatures match. If a trial was interrupted, or you change the scoring code, choose a new `SWEEP_NAME`. Avoid running the same sweep twice at the same time.

## Generation and clustering settings

| Hyperparameter | What it means and what changing it may do |
| --- | --- |
| **Guidance strength** (`sampling_config.classifier_scale`) | How strongly generated child rows follow the patterns learned for their parent's group. Higher values may improve relationships but reduce variety; zero removes the guidance gradient's contribution. |
| **Cluster count** (`clustering_config.num_clusters`) | How many groups represent different parent–child patterns. Fewer groups capture broad patterns; more groups capture finer differences but have fewer examples each. |
| **Parent-feature weighting** (`clustering_config.parent_scale`) | How much the parent's attributes influence grouping compared with the child's attributes. Increasing it emphasizes the parent; decreasing it emphasizes child behavior. |
| **Clustering method** (`clustering_config.clustering_method`) | The algorithm that creates those groups. K-means groups by similarity to a centre; GMM uses probability-based groups that can represent different shapes and spreads. |

Guidance strength can change without retraining. Changing clustering settings requires fresh clustering and retraining because the models learn from those group labels.

## Training settings

These settings apply to the diffusion model, classifier, or both, as shown by the config fields.

| Hyperparameter | What it means and what changing it may do |
| --- | --- |
| **Learning rate** (`diffusion_config.lr`, `classifier_config.lr`) | How big a change the model makes during each learning step. Too high can make learning unstable; too low can make progress slow. |
| **Training steps** (`diffusion_config.iterations`, `classifier_config.iterations`) | How many times the model updates its weights, rather than how many complete passes it makes through the data. More steps give it more opportunities to learn, but eventually improvement may stop or it may memorize examples. |
| **Network size** (`diffusion_config.d_layers`, `classifier_config.d_layers`) | The number of layers and their widths—the model's capacity. Larger networks can learn more complex patterns but require more memory and can overfit small tables. |
| **Dropout** (`diffusion_config.dropout`) | The fraction of internal units randomly switched off during diffusion-model training to discourage memorization. Some can improve generalization; too much can prevent the model from learning enough. |
| **Diffusion timesteps** (`diffusion_config.num_timesteps`) | The number of noise levels used by the diffusion process and the steps used to generate data. More timesteps make generation slower and may improve quality, but improvement is not guaranteed. |

Changing a training setting requires retraining the affected model. If the diffusion timesteps change, retrain both the diffusion model and classifier so their noise schedules remain compatible.

## Read the historical all-table comparison

| Score | What to look for |
| --- | --- |
| Foreign-key validity and primary-key uniqueness | Both should be **1.0**: links resolve and primary keys are unique. |
| Cardinality KS distance | Lower is better: the number of children per parent resembles the real data, including parents with no children. |
| Cross-table correlation RMSE | Lower is better: relationships between parent and child attributes resemble the real data. |
| Numerical KS and categorical total variation distances | Lower is better: individual column distributions resemble the real data. |
| Account-date/loan-date correlation error | Lower is better: this specific relationship is closer to the real-data relationship. |

The comparison also counts undefined correlation pairs. Missing scores are not evidence of good quality.

Use the fresh **guidance = 1** trial as the controlled baseline. The notebook explicitly puts models into evaluation mode, so older saved generations may differ even at the same guidance value.

These scores measure similarity to the training data. They do not establish held-out task performance or privacy protection, and a single seed does not establish a reliable improvement.

## Tuning order (supported by the new notebook; new training not yet run)

1. Compare guidance strengths using the existing models.
2. Explore cluster count, parent-feature weighting, and clustering method with fresh checkpoints.
3. Explore learning rates, training steps, network size, dropout, and diffusion timesteps.
4. Repeat promising configurations across seeds and check quality tradeoffs before selecting one.

## Additional controls in the tuning notebook

| Hyperparameter | Simple meaning |
| --- | --- |
| **Batch size** (`diffusion_config.batch_size`, `classifier_config.batch_size`) | Rows used in each update. Bigger batches need more memory and can change learning; at fixed steps they also process more rows, so interpret the comparison accordingly. |
| **Classifier time embedding** (`classifier_config.dim_t`) | How much space the guidance classifier uses to represent diffusion noise level. Larger values add capacity and cost; they do not necessarily improve guidance. |
| **Sampling seed** | Changes random generation while holding learned models fixed. Repeat seeds to see whether apparent gains persist. |
| **Training seed** | Changes stochastic training/clustering operations that use the global seed. Repeat separately from sampling seeds to assess model variability; toolkit-fixed internal splits remain fixed. |

Candidate values in `PRESETS` are starting experiments, not proven optimal ranges.
For a deliberate combination, change a copy of `BASE`, then vary one further
parameter. Do not launch all preset families together by default.
