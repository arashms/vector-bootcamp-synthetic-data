# Multi-table experiments — start from zero

**Start with [prepare_account_sample.ipynb](prepare_account_sample.ipynb).** No completed experiments,
trained generator checkpoints, or saved synthetic data are required. You execute
the notebooks yourself. This workflow keeps one notebook per hyperparameter and
one final comparison notebook, with shared implementation underneath.

## Run order

You execute the notebooks. No data preparation, training, generation, or notebook
execution is launched automatically by these files.

1. **Prepare the environment.** From the repository terminal run
   `uv sync --locked --group tabular-data --inexact`, then select the `.venv` kernel.
   Full raw Berka tables already available in `../data/berka/raw_data` can be reused.
   If yours live elsewhere, set the raw-data path below; do not sample the earlier
   20,000-row processed transaction file to recover histories.
2. **Create the account cohort on CPU.** Open
   [prepare_account_sample.ipynb](prepare_account_sample.ipynb). Review raw/output
   paths, **500 accounts and seed 42**, then run the preview. Set
   `PREPARE_DATASET = True` and rerun to write a separate processed dataset.
   All 8 tables are retained; every raw transaction for selected accounts is kept.
3. **Establish the baseline.** Open [00_baseline.ipynb](00_baseline.ipynb), confirm
   the selected `DATA_DIR`, preview with `RUN_NEW_TRIALS = False`, then set it to
   `True` yourself on a GPU. It trains/generates one reference and evaluates both
   full-table quality and overdraft/transaction quality. Record actual runtime
   and results in [PROGRESS.md](PROGRESS.md).
4. **Tune one parameter at a time.** Start with
   [guidance_strength_tuning.ipynb](guidance_strength_tuning.ipynb); it reuses the
   baseline models. Keep the same input dataset/base settings, narrow `VALUES`,
   preview, then enable generation when ready. Each parameter value is compared
   with its included baseline value. Repeat promising settings across seeds later.
5. **Compare completed experiments on CPU.** Run
   [all_experiments.ipynb](all_experiments.ipynb). It discovers this dataset's
   completed runs and evaluates them together with one detector. Inspect both
   metric groups and the per-table/per-edge CSVs before filling the conclusions.

Use one generating notebook at a time. The first full baseline training duration
has not been measured. The 500-account raw preview contains 117,560 transactions
and 247 overdraft rows for the currently available source and seed, considerably
more transactions than the previous row sample. Full histories, larger models,
and more candidates increase generation cost. Read actual preview counts rather
than assuming those numbers for a different raw source.

## Change the account count or dataset path

Edit [dataset_settings.py](dataset_settings.py), then **restart notebook kernels**:

```python
ACCOUNT_SAMPLE_SIZE = 500  # None means all accounts.
DATA_SEED = 42
RAW_DATA_DIR = MULTI_TABLE / "data/berka/raw_data"
COHORT_LABEL = "all" if ACCOUNT_SAMPLE_SIZE is None else str(ACCOUNT_SAMPLE_SIZE)
DATA_DIR = MULTI_TABLE / "data" / f"berka_accounts_{COHORT_LABEL}_seed{DATA_SEED}"
```

The default `COHORT_LABEL` is derived from account count. Changing count/seed in
that file derives a new processed directory. You can instead set `DATA_DIR` to
any existing **Berka-compatible processed dataset** (for example,
`DATA_DIR = Path("/path/to/processed_dataset")`) with the 8 CSVs, domain files,
`dataset_meta.json`, and `preprocess_meta.json`. For a different schema, adapting
its table/ID rules and task evaluator is separate work.

The preparation notebook exposes `RAW_DATA_DIR`, `ACCOUNT_SAMPLE_SIZE`,
`SAMPLING_SEED`, and `OUTPUT_DATA_DIR`; every experiment exposes
`DATA_DIR = DEFAULT_DATA_DIR`. If overriding values locally, update the shared
`DATA_DIR` or override it consistently in every notebook you run. Always create
that dataset's matching baseline before its parameter sweeps.

Different data contents produce distinct cache/report fingerprints even under a
common experiment root. Discovery filters out other datasets. Evaluation rejects
trial/reference fingerprint mismatches. Identical contents may reuse verified
stages. Existing datasets and synthesized files are not deleted or replaced.

## Account sampling and preservation

Select account IDs uniformly with a fixed seed, without using overdraft labels.
Retain their `trans`, `loan`, `order`, and `disp` rows; cards follow dispositions,
clients follow dispositions, and districts are the union referenced by accounts
and clients. A shared client does not pull other accounts into the cohort. This
preserves complete **observed raw histories for selected accounts**, rather than
claiming complete histories for every client. All 8 declared relationships must
resolve.

The helper reuses existing preprocessing functions and writes a separate dataset:
CSV/domain files, codebooks, date epochs, `selected_accounts.csv`, and an
`account_sample_manifest.json` recording raw/source/config fingerprints and
artifact checksums. `preprocess_meta.json` records the sampling unit and counts.
There is no transaction-row limit in this mode.

Only an identical completed output is reused after verification. Existing unrelated,
incomplete, differently configured, or modified outputs are rejected; choose a new
output directory. Empty child tables are rejected; preview small-table and event
counts before training. Small cohorts can make clustering and evaluation noisy.

## Your first baseline

The baseline copies the original YAML settings and aligns diffusion/classifier
internal splits to **70/20/10**. They fit preprocessing independently, so using the
same split avoids training guidance in a different feature coordinate system.
The original diffusion config used 99/0.5/0.5 while its classifier defaulted to
70/20/10. This workflow establishes its own fresh reference.

| Setting | Initial value |
| --- | --- |
| Training / sampling seed | 42 / 42 |
| Guidance strength | 1.0 |
| Clustering | 50 groups, parent scale 1.0, K-means + GMM |
| Training / generation | Original YAML model settings; full sample scale 1.0 |
| Internal model split | 70% train, 20% validation, 10% test for both models |

Keep this baseline fixed while varying one parameter. If you intentionally change
other base settings, use the same revised baseline in `00_baseline.ipynb` and the
parameter notebooks. The parameter notebooks require a completed matching baseline
before enabling a sweep. No previous findings or preferred settings are assumed.

## Notebook index

### Start, inspect, and compare

| Notebook | When to use it |
| --- | --- |
| [prepare_account_sample.ipynb](prepare_account_sample.ipynb) | First: preview and prepare the 500-account cohort. |
| [00_baseline.ipynb](00_baseline.ipynb) | Then: train, generate, and evaluate the baseline. |
| [transaction_overdraft_evaluation.ipynb](transaction_overdraft_evaluation.ipynb) | After generation: detailed transaction/overdraft diagnostics for your completed runs. |
| [all_experiments.ipynb](all_experiments.ipynb) | After parameter runs: compare configurations and their outcomes together. |
| [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb) | Optional generic template for choosing a parameter family. |

### Guidance and clustering

| Notebook | Parameter |
| --- | --- |
| [guidance_strength_tuning.ipynb](guidance_strength_tuning.ipynb) | Guidance strength; reuse your new baseline models. |
| [cluster_count.ipynb](cluster_count.ipynb) | Number of parent/child groups. |
| [parent_weight.ipynb](parent_weight.ipynb) | Parent-feature influence on grouping. |
| [clustering_method.ipynb](clustering_method.ipynb) | Grouping algorithm. |

### Diffusion training

| Notebook | Parameter |
| --- | --- |
| [diffusion_learning_rate.ipynb](diffusion_learning_rate.ipynb) | Update size. |
| [diffusion_training_steps.ipynb](diffusion_training_steps.ipynb) | Number of updates. |
| [diffusion_network_size.ipynb](diffusion_network_size.ipynb) | Hidden layers and widths. |
| [diffusion_dropout.ipynb](diffusion_dropout.ipynb) | Units randomly disabled during training. |
| [diffusion_timesteps.ipynb](diffusion_timesteps.ipynb) | Noise levels and generation steps. |
| [diffusion_batch_size.ipynb](diffusion_batch_size.ipynb) | Rows per update. |

### Generator's guidance classifier

These predict parent groups for generation. The separate overdraft evaluator
stays fixed across trials.

| Notebook | Parameter |
| --- | --- |
| [classifier_learning_rate.ipynb](classifier_learning_rate.ipynb) | Update size. |
| [classifier_training_steps.ipynb](classifier_training_steps.ipynb) | Number of updates. |
| [classifier_network_size.ipynb](classifier_network_size.ipynb) | Hidden layers and widths. |
| [classifier_batch_size.ipynb](classifier_batch_size.ipynb) | Rows per update. |
| [classifier_time_embedding.ipynb](classifier_time_embedding.ipynb) | Representation of the noise level. |

Every parameter notebook follows **settings → preview → generation → evaluation
→ conclusion**. It includes baseline and seed controls. Completed stages are
reused, and repeated candidate values are deduplicated. Conclusions are blank
until you run the experiment.

The final notebook compares completed runs. A combined setting needs a deliberate
new configuration, its own generated run, and evaluation; it is not trained
by the final comparison notebook.

## Evaluation in simple terms

**Every baseline, parameter notebook, and final comparison runs both groups below.**
No universal weighted score or automatic winner combines them.

### Full multi-table quality

| Check | What it tells you |
| --- | --- |
| Column KS / categorical TVD on all 8 tables | Whether each table's distributions resemble the selected real cohort. |
| Missing/invalid and unknown-category rates | Whether data has unsupported or missing values hidden by other scores. |
| Primary/foreign keys on all tables/edges | Whether identities are unique/present and relationships resolve. |
| Children per parent, including zeros | Whether relationship sizes resemble the cohort. |
| Within-table and parent/child mixed correlations | Whether feature associations are retained; constant/undefined pairs are counted. |
| Client/account combinations through `disp` | Whether matched parent attributes retain their joint associations. Report mixed-correlation error and joint categorical TVD. |

Marginal summaries average table means equally; the transaction table does not
outweigh other tables merely by having more rows. Eligible missing column/edge
scores propagate as missing. Correlation RMSE uses defined pairs and reports
undefined-pair counts alongside it. Per-table/edge scores remain visible.

IDs never enter numerical fidelity/correlation measurements, including external
`trans.account` and `order.account_to`. They are used for integrity and joins as
appropriate. All comparisons use the selected processed cohort, not the full raw
population. Row-count deviations are diagnostics, not exact equality constraints
on stochastic generation.

### Transaction and overdraft quality

The supplied random forest learns from real transactions and is tested on
separate real accounts and each synthetic run. Its target is **current negative
balance** (`balance < 0`). Balance, target-derived features, and IDs are excluded
from its predictors. No loan classification is used. Guidance-classifier tuning
changes generator classifiers, not this fixed task-evaluation model.

Report AP, ROC-AUC, fixed-threshold precision/recall, event counts, and prevalence.
Standardized AP controls the class-prior difference only. Account-bootstrap
intervals hold the fitted detector and saved generation fixed; they do not replace
repeated generation/training seeds.

Transaction diagnostics include overall/overdraft-conditional distributions,
category/monthly event rates with support, account counts/volumes, integrity, and
chronology after aligning date epochs. Complete raw account histories now permit
better source diagnostics; the row-based relational generator still does not
guarantee faithful sequences. Future rolling/history features need a separate,
leakage-safe design.

The detector's held-out real accounts are not excluded before generator training.
The agreed study is **real→synthetic transfer plus fidelity**, not synthetic-trained
utility on an independent real holdout or privacy protection. No extra outer holdout
or synthetic→real classifier has been added.

## Shared code and saved artifacts

| Change | Work repeated after the baseline |
| --- | --- |
| Guidance or sampling seed | Generation only. |
| Guidance-classifier settings | Classifier training and generation; reuse diffusion/clustering. |
| Diffusion settings | Diffusion, classifier training, and generation; reuse clustering. |
| Clustering or training seed | Clustering and all downstream stages. |

`dataset_settings.py` selects shared data paths.
`../data_preprocessing/account_sampling.py` prepares account cohorts using the
existing preprocessing transforms.
`tuning.py` reuses toolkit functions for training/sampling and controls caching.
`../evaluation/overdraft_transfer_evaluator.py` supplies the detector;
`transaction_quality.py`, `multi_table_quality.py`, and `transaction_evaluation.py` provide shared diagnostics
and report saving. Each notebook stays an experimental interface to these helpers.

All new notebooks share paths defined in `tuning.py`:

```text
../results/experiments/transaction_tuning_from_scratch/
    clustering/    diffusion/    models/    samples/
../results/experiments/transaction_overdraft_from_scratch/<report-fingerprint>/
```

These directories do not import older runs or original model checkpoints.
Manifests identify data, relevant configs, seeds, source/dependency fingerprints,
and device. Status files record duration, completion/failure, and artifact hashes.
Complete compatible stages are reused. Partial runs are excluded from evaluation;
changed completed artifacts are rejected. Failed stages need inspection and a
fresh root to retry; there is no optimizer-state resume.

If you change the shared root, change it consistently for the baseline, parameter,
and comparison notebooks. Generation uses the actual matched tables captured
before toolkit cleanup drops keys. Reports save split IDs, predictions, detailed
CSVs, settings/source fingerprints, uncertainty, and plots.

Seeds control stochastic operations using them; some toolkit internal splits and
random states are fixed. Reusing a cache does not change subsequent per-relation
training draws. Cross-hardware determinism is not guaranteed.

## Validation and reference files

No real-cohort preparation, model training, synthetic generation, or notebook execution is performed automatically for this
handoff. The new workflow is ready for your first run; full fresh generator
execution is still unvalidated. Inspect the baseline's actual results before
launching a broad sweep. Development checks and decisions are in [PROGRESS.md](PROGRESS.md).

Targeted checks from the repository root:

```bash
.venv/bin/python -m pytest -q implementations/tabular_data/multi_table/tests/
```

[guidance_strength.ipynb](guidance_strength.ipynb) remains a historical notebook.
It is outside the new run order. Original preprocessing/training/synthesis files
remain reference implementations.

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
