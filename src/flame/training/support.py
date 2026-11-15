"""Fits the k-NN support index (one per fuel) at export time. Querying it (query_support, confidence_level)
lives in inference.py, which engine copies verbatim."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from configs.training.config_training import Support


def fit_partition(values: np.ndarray, reference_rows: np.ndarray, support: Support) -> dict[str, Any]:
    scaler = StandardScaler().fit(values)
    neighbors = NearestNeighbors(n_neighbors=min(support.nearest_k, len(values))).fit(scaler.transform(values))
    nearest_other_test, _ = neighbors.kneighbors(n_neighbors=1)
    return {
        "scaler": scaler,
        "neighbors": neighbors,
        "threshold": float(np.percentile(nearest_other_test[:, 0], support.range_percentile)),
        "reference_rows": np.asarray(reference_rows),
    }


def fit_support(features: pd.DataFrame, support: Support) -> dict[str, Any]:
    columns = list(support.columns)
    partitions = {}
    for value, members in features.groupby(support.partition_column, sort=True):
        partitions[str(value)] = fit_partition(members[columns].to_numpy(dtype=float), members.index.to_numpy(),
                                               support)
    return {
        "columns": columns,
        "partition_column": support.partition_column,
        "range_percentile": support.range_percentile,
        "nearest_k": support.nearest_k,
        "partitions": partitions,
    }


def support_thresholds(bundle: dict[str, Any]) -> dict[str, float]:
    return {name: partition["threshold"] for name, partition in bundle["partitions"].items()}
