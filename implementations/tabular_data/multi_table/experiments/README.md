# Multi-table hyperparameter tuning

Start with [guidance_strength.ipynb](guidance_strength.ipynb). It reuses the trained ClavaDDPM models to explore guidance strength without retraining.

Hyperparameters are settings we choose to control how the model groups, learns, or generates data. The goal is to improve relationships between tables while keeping individual columns realistic and preserving data diversity.

## What is available now

- One notebook for guidance-strength experiments, with separate settings, generated data, and scores for each trial.
- Evaluation of all eight parent–child relationships, column distributions, and final matched tables with their keys.
- The existing training and synthesis notebooks, YAML configs, and baseline results remain unchanged by these experiments.

The other hyperparameters below are **planned experiments**. The current notebook only sweeps guidance strength.

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

## Read the comparison

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

## Planned tuning order

1. Compare guidance strengths using the existing models.
2. Explore cluster count, parent-feature weighting, and clustering method with fresh checkpoints.
3. Explore learning rates, training steps, network size, dropout, and diffusion timesteps.
4. Repeat promising configurations across seeds and check quality tradeoffs before selecting one.
