"""Run ClavaDDPM training and synthesis from YAML configs without notebooks.

Example:
    uv run --group tabular-data python run_clavaddpm_experiment.py \
        --base-data-dir /path/to/berka_feature_engineered \
        --training-config ../training/config.yaml \
        --synthesizing-config ../synthesizing/config.yaml \
        --trans-sample-size 100000 \
        --output-dir /path/to/results/feature_100k_run1
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import yaml
from midst_toolkit.common.config import (
    ClavaDDPMClassifierConfig,
    ClavaDDPMClusteringConfig,
    ClavaDDPMDiffusionConfig,
    ClavaDDPMMatchingConfig,
    ClavaDDPMSamplingConfig,
    GeneralConfig,
)
from midst_toolkit.common.logger import TOOLKIT_LOGGER
from midst_toolkit.common.random import set_all_random_seeds
from midst_toolkit.common.variables import DEVICE
from midst_toolkit.models.clavaddpm.clustering import clava_clustering
from midst_toolkit.models.clavaddpm.data_loaders import Table, load_tables
from midst_toolkit.models.clavaddpm.synthesizer import clava_synthesizing
from midst_toolkit.models.clavaddpm.train import clava_training


LOGGER = logging.getLogger("clavaddpm_experiment")


def _read_yaml(path: Path, label: str) -> dict[str, Any]:
    """Load a mapping from a YAML file."""
    if not path.is_file():
        raise FileNotFoundError(f"{label} config does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"{label} config must contain a YAML mapping: {path}")
    return config


def _validate_config_sections(
    training_config: dict[str, Any],
    synthesizing_config: dict[str, Any],
) -> None:
    """Check that the config files contain the sections used by ClavaDDPM."""
    required_training = {"clustering_config", "diffusion_config", "classifier_config"}
    missing_training = required_training.difference(training_config)
    if missing_training:
        raise ValueError(f"Training config is missing sections: {sorted(missing_training)}")

    required_synthesizing = {"sampling_config", "matching_config"}
    missing_synthesizing = required_synthesizing.difference(synthesizing_config)
    if missing_synthesizing:
        raise ValueError(f"Synthesizing config is missing sections: {sorted(missing_synthesizing)}")


def _sample_transactions(
    tables: dict[str, Table],
    sample_size: int | None,
    seed: int,
) -> tuple[int, int]:
    """Subsample trans consistently in model and original-data frames."""
    if sample_size is not None and (
        not isinstance(sample_size, int) or isinstance(sample_size, bool) or sample_size <= 0
    ):
        raise ValueError("trans_sample_size must be a positive integer or None.")
    if "trans" not in tables:
        raise ValueError("The loaded ClavaDDPM dataset does not contain a 'trans' table.")

    trans_table = tables["trans"]
    original_rows = len(trans_table.data)
    if len(trans_table.original_data) != original_rows:
        raise ValueError(
            "The trans model and original-data frames have different row counts; "
            "refusing to sample them inconsistently.",
        )
    if sample_size is None or sample_size >= original_rows:
        return original_rows, original_rows

    selected_indices = trans_table.data.sample(n=sample_size, random_state=seed).index
    trans_table.data = trans_table.data.loc[selected_indices].reset_index(drop=True)
    trans_table.original_data = trans_table.original_data.loc[selected_indices].reset_index(drop=True)
    return original_rows, len(trans_table.data)


def run_experiment(  # noqa: PLR0915
    *,
    base_data_dir: Path,
    training_config_path: Path,
    synthesizing_config_path: Path,
    output_dir: Path,
    trans_sample_size: int | None,
    exp_name: str = "ClavaDDPM_experiment",
    sample_prefix: str = "synthetic",
    dry_run: bool = False,
) -> Path:
    """Train and synthesize one isolated ClavaDDPM experiment.

    ``trans_sample_size`` samples rows from the preprocessed ``trans`` table
    loaded from ``base_data_dir``. Other tables stay intact, so account foreign
    keys remain valid. ``None`` uses every transaction row.

    Each experiment must use a new ``output_dir``. This prevents ClavaDDPM from
    silently reusing clustering checkpoints or overwriting another run's model
    artifacts and synthetic output.
    """
    base_data_dir = base_data_dir.expanduser().resolve()
    training_config_path = training_config_path.expanduser().resolve()
    synthesizing_config_path = synthesizing_config_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not base_data_dir.is_dir():
        raise NotADirectoryError(f"ClavaDDPM base data directory does not exist: {base_data_dir}")
    if output_dir.exists():
        raise FileExistsError(
            f"Experiment output directory already exists: {output_dir}. "
            "Choose a new directory so prior checkpoints are not reused or overwritten.",
        )
    if not exp_name.strip() or Path(exp_name).name != exp_name:
        raise ValueError("exp_name must be a non-empty single directory name.")
    if not sample_prefix.strip() or Path(sample_prefix).name != sample_prefix:
        raise ValueError("sample_prefix must be a non-empty single directory name.")

    training_config = _read_yaml(training_config_path, "Training")
    synthesizing_config = _read_yaml(synthesizing_config_path, "Synthesizing")
    _validate_config_sections(training_config, synthesizing_config)

    training_seed = training_config.get("random_seed", 42)
    if training_seed is not None and not isinstance(training_seed, int):
        raise ValueError("training random_seed must be an integer or null.")
    synthesizing_seed = synthesizing_config.get("random_seed", training_seed)
    if synthesizing_seed is not None and not isinstance(synthesizing_seed, int):
        raise ValueError("synthesizing random_seed must be an integer or null.")
    sample_seed = 42 if training_seed is None else training_seed

    clustering_config = ClavaDDPMClusteringConfig(**training_config["clustering_config"])
    diffusion_config = ClavaDDPMDiffusionConfig(**training_config["diffusion_config"])
    classifier_config = ClavaDDPMClassifierConfig(**training_config["classifier_config"])
    sampling_config = ClavaDDPMSamplingConfig(**synthesizing_config["sampling_config"])
    matching_config = ClavaDDPMMatchingConfig(**synthesizing_config["matching_config"])
    sample_scale = synthesizing_config["sampling_config"].get("sample_scale", 1.0)
    if not isinstance(sample_scale, (int, float)) or isinstance(sample_scale, bool) or sample_scale < 0:
        raise ValueError("sampling_config.sample_scale must be a non-negative number.")

    TOOLKIT_LOGGER.setLevel(logging.INFO)
    if training_seed is not None:
        set_all_random_seeds(seed=training_seed)

    LOGGER.info("Loading ClavaDDPM data from %s", base_data_dir)
    tables, relation_order, _ = load_tables(base_data_dir)
    rows_before, rows_after = _sample_transactions(tables, trans_sample_size, sample_seed)
    LOGGER.info("Transaction rows: %s -> %s", f"{rows_before:,}", f"{rows_after:,}")

    experiment_metadata = {
        "base_data_dir": str(base_data_dir),
        "trans_rows_before_sampling": rows_before,
        "trans_rows_used_for_training": rows_after,
        "trans_sample_size": trans_sample_size,
        "training_seed": training_seed,
        "transaction_sample_seed": sample_seed,
        "synthesizing_seed": synthesizing_seed,
        "exp_name": exp_name,
        "sample_prefix": sample_prefix,
        "training_config": training_config,
        "synthesizing_config": synthesizing_config,
    }

    if dry_run:
        LOGGER.info("Dry run complete; data and configs validated. No output directory created.")
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()
    try:
        with (output_dir / "training_config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(training_config, file, sort_keys=False)
        with (output_dir / "synthesizing_config.yaml").open("w", encoding="utf-8") as file:
            yaml.safe_dump(synthesizing_config, file, sort_keys=False)
        with (output_dir / "experiment.json").open("w", encoding="utf-8") as file:
            json.dump(experiment_metadata, file, indent=2)
            file.write("\n")

        LOGGER.info("Clustering tables...")
        clustered_tables, all_group_lengths_prob_dicts = clava_clustering(
            tables,
            relation_order,
            output_dir,
            clustering_config,
        )

        LOGGER.info("Training ClavaDDPM models on %s", DEVICE)
        _, models = clava_training(
            clustered_tables,
            relation_order,
            output_dir,
            diffusion_config,
            classifier_config,
            device=DEVICE,
        )
        if synthesizing_seed is not None:
            set_all_random_seeds(seed=synthesizing_seed)

        LOGGER.info("Synthesizing tables...")
        general_config = GeneralConfig(
            data_dir=base_data_dir,
            test_data_dir=base_data_dir,
            workspace_dir=output_dir,
            exp_name=exp_name,
            sample_prefix=sample_prefix,
        )
        clava_synthesizing(
            clustered_tables,
            relation_order,
            output_dir,
            models,
            general_config,
            sampling_config,
            matching_config,
            all_group_lengths_prob_dicts,
            sample_scale=sample_scale,
        )
        LOGGER.info(
            "Experiment complete. Synthetic transaction CSV: %s",
            output_dir / exp_name / "trans" / f"{sample_prefix}_final" / "trans_synthetic.csv",
        )
    except Exception:
        LOGGER.exception("ClavaDDPM experiment failed; partial outputs remain in %s", output_dir)
        raise

    return output_dir


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data-dir", type=Path, required=True, help="Preprocessed ClavaDDPM input directory.")
    parser.add_argument("--training-config", type=Path, required=True, help="ClavaDDPM training YAML config.")
    parser.add_argument("--synthesizing-config", type=Path, required=True, help="ClavaDDPM synthesizing YAML config.")
    parser.add_argument(
        "--trans-sample-size",
        type=int,
        default=None,
        help="Randomly sample this many preprocessed trans rows; omit to use all rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, unique directory for this experiment's configs, checkpoints, and synthetic tables.",
    )
    parser.add_argument("--exp-name", default="ClavaDDPM_experiment", help="Synthesis subdirectory name.")
    parser.add_argument("--sample-prefix", default="synthetic", help="Synthetic output prefix.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configs and data loading/sampling without training or creating outputs.",
    )
    return parser.parse_args()


def main() -> None:
    """Run one experiment from CLI arguments."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run_experiment(
        base_data_dir=args.base_data_dir,
        training_config_path=args.training_config,
        synthesizing_config_path=args.synthesizing_config,
        output_dir=args.output_dir,
        trans_sample_size=args.trans_sample_size,
        exp_name=args.exp_name,
        sample_prefix=args.sample_prefix,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
