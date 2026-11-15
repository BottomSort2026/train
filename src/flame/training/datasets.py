from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from flame.training import flex_data
from flame.training.config import ExperimentConfig, resolve_path


@dataclass(frozen=True)
class TrainingData:
    features: pd.DataFrame
    target: pd.Series
    groups: pd.Series
    rows: pd.DataFrame
    exclusions: dict[str, int]
    source_rows: int
    source_sha256: str
    id_columns: list[str]
    row_key: str

    @property
    def ids(self) -> pd.DataFrame:
        return self.rows[self.id_columns]

    @property
    def row_keys(self) -> pd.Series:
        return self.rows[self.row_key]


def exclusion_mask(column: pd.Series, rule: str, value: Any = None) -> pd.Series:
    if rule == "not_null":
        return column.notna()
    if rule == "is_null":
        return column.isna()
    if rule == "is_true":
        return column.fillna(False).astype(bool)
    if rule == "equals":
        return column == value
    if rule == "not_equals":
        return column != value
    raise ValueError(f"Unknown exclusion rule: {rule!r}")


def apply_exclusions(table: pd.DataFrame, exclusions: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, int]]:
    kept = table
    removed: dict[str, int] = {}
    for exclusion in exclusions:
        mask = exclusion_mask(kept[exclusion["column"]], exclusion["rule"], exclusion.get("value"))
        removed[exclusion["name"]] = int(mask.sum())
        kept = kept[~mask]
    return kept.reset_index(drop=True), removed


def build_target(rows: pd.DataFrame, config: ExperimentConfig) -> pd.Series:
    target = config.section("target")
    values = rows[target["column"]]
    if config.task == "classification":
        mapped = values.map(target["class_map"])
        if mapped.isna().any():
            unmapped = sorted(values[mapped.isna()].astype(str).unique())
            raise ValueError(f"Target values without a class in target.class_map: {unmapped}")
        return mapped.rename("target")
    if values.isna().any():
        raise ValueError(f"Target column {target['column']!r} has missing values after the exclusions")
    return values.astype(float).rename("target")


def build_features(rows: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    spec = config.section("features")
    features = rows[config.feature_columns].copy()
    for column in spec["categorical"]:
        features[column] = features[column].astype(str)
    for column in spec["numeric"]:
        features[column] = features[column].astype(float)
    for column in spec["boolean"]:
        features[column] = features[column].astype(bool)
    if features.isna().any().any():
        raise ValueError("Feature columns contain missing values after the exclusions")
    return features


def build_training_data(table: pd.DataFrame, config: ExperimentConfig, source_sha256: str) -> TrainingData:
    dataset = config.section("dataset")
    exclusions = [*dataset["exclusions"], *dataset.get("extra_exclusions", [])]
    rows, removed = apply_exclusions(table, exclusions)
    return TrainingData(
        features=build_features(rows, config),
        target=build_target(rows, config),
        groups=rows[dataset["group_column"]].astype(str).rename("group"),
        rows=rows,
        exclusions=removed,
        source_rows=int(len(table)),
        source_sha256=source_sha256,
        id_columns=list(dataset["id_columns"]),
        row_key=dataset["row_key"],
    )


def load_source_table(config: ExperimentConfig) -> tuple[pd.DataFrame, str]:
    path = resolve_path(config.section("dataset")["path"])
    table = flex_data.load_flex(path, verify=True)
    manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    return table, manifest["sha256"]


def load_training_data(config: ExperimentConfig) -> TrainingData:
    table, source_sha256 = load_source_table(config)
    return build_training_data(table, config, source_sha256)
