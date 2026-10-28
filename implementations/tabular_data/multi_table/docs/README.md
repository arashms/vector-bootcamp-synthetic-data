# Multi-table overdraft diagram

[PNG](solution_architecture.png) · [Editable SVG](solution_architecture.svg)

The diagram follows the supplied product, roadmap, and Build Day 1 slides.
It emphasizes the team's tuning, evaluation, graph analysis, and target-context
feature work while keeping the supplied generator backbone at a high level.

Day 1 work includes the real-trained overdraft detector, comparison against a
real holdout and two synthetic generations, distribution and category-rate
diagnostics, a larger transaction sample, transaction graph analysis, and the
start of a five-strength guidance sweep with trained-model reuse. Quality checks
cover all eight parent–child relationships.

Clustering/training refinements, enriched features, and challenger comparisons
are labeled as next/planned. The three feature families are candidate additions;
history-derived features require suitable histories and prior information only.
Graph-based privacy metrics remain future work.

Challenger descriptions were checked against primary sources:

- [REaLTabFormer: World Bank repository](https://github.com/worldbank/REaLTabFormer)
  uses autoregressive and sequence-to-sequence transformers for tabular and
  relational generation.
- [RelDiff: authors' paper](https://arxiv.org/abs/2506.00710) models the foreign-key
  graph and uses graph-conditioned diffusion for relational attributes.
  [Official repository](https://github.com/ValterH/RelDiff).

The diagram proposes comparisons; it does not claim either challenger has been
integrated or evaluated in this project. The SVG model names link to the sources.

Regenerate the 3200 × 1800 PNG and SVG from the repository root:

```bash
MPLCONFIGDIR=/tmp/multi-table-mpl .venv/bin/python implementations/tabular_data/multi_table/docs/render_solution_architecture.py
```
