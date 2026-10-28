"""Evaluate real-trained overdraft detection on ClavaDDPM transaction datasets.

Inputs must be preprocessed transaction tables with the same categorical codes
and date representation as ClavaDDPM's ``trans.csv`` output. Example:

    evaluator = OverdraftTransferEvaluator("data/berka2/trans.csv")
    evaluator.evaluate_synthetic(
        "results2/.../trans_synthetic.csv",
        name="run-2",
        config={"trans_sample_size": 100000},
    )
    print(evaluator.get_metrics())
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = ["trans_date", "amount"]
CATEGORICAL_FEATURES = ["trans_type", "operation", "k_symbol", "bank"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "overdraft"
ACCOUNT_COLUMN = "account_id"
DEFAULT_RANDOM_FOREST_PARAMS: dict[str, Any] = {
    "n_estimators": 150,
    "max_depth": 14,
    "min_samples_leaf": 10,
    "class_weight": "balanced_subsample",
    "max_samples": 0.5,
    "n_jobs": -1,
    "random_state": 42,
}


def _read_frame(data: pd.DataFrame | str | Path, dataset_name: str) -> pd.DataFrame:
    """Return a copy of a dataframe or read a transaction CSV."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    path = Path(data)
    if not path.is_file():
        raise FileNotFoundError(f"{dataset_name} data file does not exist: {path}")
    return pd.read_csv(path)


class OverdraftTransferEvaluator:
    """Train on account-held-out real data and evaluate named synthetic datasets.

    The real and synthetic tables should use the preprocessed schema produced
    for ClavaDDPM: ``account_id``, ``trans_date``, ``trans_type``, ``operation``,
    ``amount``, ``balance``, ``k_symbol``, and ``bank``. The target is derived
    from ``balance < 0``; balance and IDs are never used as predictors.

    Categorical values must share the same code mapping in real and synthetic
    inputs. Generated categorical floats are rounded to the nearest valid code;
    out-of-domain values become ``__unknown__``. Use ``name`` when evaluating
    a synthetic dataset to identify its ClavaDDPM hyperparameter configuration.
    """

    def __init__(
        self,
        real_data: pd.DataFrame | str | Path,
        *,
        test_size: float = 0.2,
        random_state: int = 42,
        threshold: float = 0.5,
        random_forest_params: dict[str, Any] | None = None,
        real_data_config: Mapping[str, Any] | None = None,
    ) -> None:
        """Split and fit real data, retaining its ClavaDDPM input configuration."""
        if not 0 < test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1; received {test_size}.")
        if not 0 <= threshold <= 1:
            raise ValueError(f"threshold must be between 0 and 1; received {threshold}.")

        self.threshold = threshold
        self.random_state = random_state
        self.real_data_config = json.dumps(real_data_config or {}, sort_keys=True, default=str)
        self.real_data = self._prepare_data(_read_frame(real_data, "Real"), "Real", require_accounts=True)
        self.synthetic_data: dict[str, pd.DataFrame] = {}
        self._synthetic_metrics: dict[str, dict[str, Any]] = {}

        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_indices, test_indices = next(
            splitter.split(
                self.real_data,
                self.real_data[TARGET],
                groups=self.real_data[ACCOUNT_COLUMN],
            ),
        )
        self.real_train = self.real_data.iloc[train_indices].copy()
        self.real_test = self.real_data.iloc[test_indices].copy()
        if self.real_train[TARGET].nunique() != 2 or self.real_test[TARGET].nunique() != 2:
            raise ValueError(
                "The account-grouped real split must contain both overdraft classes in train and test. "
                "Try another random_state or test_size.",
            )

        params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
        if random_forest_params:
            params.update(random_forest_params)
        self.model = self._build_model(params)
        self.model.fit(self.real_train[FEATURES], self.real_train[TARGET])
        self.real_test_metrics = self._evaluate(
            self.real_test,
            self.model.predict_proba(self.real_test[FEATURES])[:, 1],
        )

    @staticmethod
    def _build_model(params: dict[str, Any]) -> Pipeline:
        """Build the notebook's imputation, encoding, and forest pipeline."""
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ],
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, NUMERIC_FEATURES),
                ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ],
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", RandomForestClassifier(**params)),
            ],
        )

    def _prepare_data(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        *,
        require_accounts: bool,
    ) -> pd.DataFrame:
        """Validate columns, derive labels, and align generated categorical codes."""
        required = set(FEATURES + ["balance"])
        if require_accounts:
            required.add(ACCOUNT_COLUMN)
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"{dataset_name} data is missing required columns: {sorted(missing)}")

        prepared = data.copy()
        for column in NUMERIC_FEATURES + ["balance"]:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        if prepared["balance"].isna().any():
            raise ValueError(f"{dataset_name} balance contains missing or non-numeric values.")
        if require_accounts and prepared[ACCOUNT_COLUMN].isna().any():
            raise ValueError("Real data contains missing account IDs; account-wise splitting is not possible.")

        if require_accounts:
            prepared[TARGET] = prepared["balance"].lt(0).astype("int8")
            for column in CATEGORICAL_FEATURES:
                prepared[column] = prepared[column].fillna("__missing__").astype(str)
        else:
            for column in CATEGORICAL_FEATURES:
                real_codes = pd.to_numeric(self.real_data[column], errors="coerce")
                synthetic_codes = pd.to_numeric(prepared[column], errors="coerce")
                is_code_domain = real_codes.notna().all()
                if is_code_domain:
                    valid_codes = set(real_codes.unique())
                    in_domain = (
                        synthetic_codes.notna()
                        & np.isfinite(synthetic_codes)
                        & synthetic_codes.ge(min(valid_codes) - 0.5)
                        & synthetic_codes.le(max(valid_codes) + 0.5)
                    )
                    rounded = np.floor(synthetic_codes.loc[in_domain] + 0.5)
                    valid_rounded = rounded.isin(valid_codes)
                    aligned = pd.Series("__unknown__", index=prepared.index, dtype="object")
                    aligned.loc[rounded.index[valid_rounded]] = (
                        rounded.loc[valid_rounded].astype("int64").astype(str)
                    )
                    prepared[column] = aligned
                else:
                    prepared[column] = prepared[column].fillna("__missing__").astype(str)
            prepared[TARGET] = prepared["balance"].lt(0).astype("int8")

        if prepared.empty:
            raise ValueError(f"{dataset_name} data contains no rows.")
        return prepared

    def _evaluate(self, data: pd.DataFrame, probabilities: np.ndarray) -> dict[str, Any]:
        """Calculate rare-event and fixed-threshold classification metrics."""
        labels = data[TARGET].to_numpy(dtype="int8")
        predictions = (probabilities >= self.threshold).astype("int8")
        has_both_classes = np.unique(labels).size == 2
        return {
            "rows": len(data),
            "overdrafts": int(labels.sum()),
            "overdraft_prevalence": float(labels.mean()),
            "average_precision": (
                float(average_precision_score(labels, probabilities)) if has_both_classes else np.nan
            ),
            "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both_classes else np.nan,
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "threshold": self.threshold,
        }

    def evaluate_synthetic(
        self,
        synthetic_data: pd.DataFrame | str | Path,
        *,
        name: str,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate one synthetic dataset and retain its metrics for comparison.

        ``name`` should identify the ClavaDDPM run or hyperparameter set, for
        example ``"steps=10000, batch_size=256"``. Reusing a name replaces that
        run's stored data and metrics. ``config`` is recorded with the results
        for comparison; it does not change the already-trained detector.
        """
        if not name.strip():
            raise ValueError("name must be a non-empty dataset or configuration label.")
        raw_data = _read_frame(synthetic_data, name)
        prepared = self._prepare_data(raw_data, name, require_accounts=False)
        probabilities = self.model.predict_proba(prepared[FEATURES])[:, 1]
        metrics = self._evaluate(prepared, probabilities)
        metrics["clavaddpm_config"] = json.dumps(config or {}, sort_keys=True, default=str)
        metrics["real_data_config"] = self.real_data_config
        self.synthetic_data[name] = prepared
        self._synthetic_metrics[name] = metrics
        return metrics.copy()

    def get_metrics(self) -> pd.DataFrame:
        """Return synthetic metrics for every dataset evaluated by this instance."""
        if not self._synthetic_metrics:
            return pd.DataFrame(
                columns=[
                    "rows",
                    "overdrafts",
                    "overdraft_prevalence",
                    "average_precision",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                    "threshold",
                    "clavaddpm_config",
                    "real_data_config",
                ],
            )
        return pd.DataFrame.from_dict(self._synthetic_metrics, orient="index").rename_axis("dataset")
