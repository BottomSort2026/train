from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def fit_partition(values: np.ndarray, reference_rows: np.ndarray, support: dict[str, Any]) -> dict[str, Any]:
    scaler = StandardScaler().fit(values)
    neighbors = NearestNeighbors(n_neighbors=min(support["nearest_k"], len(values))).fit(scaler.transform(values))
    nearest_other_test, _ = neighbors.kneighbors(n_neighbors=1)
    return {
        "scaler": scaler,
        "neighbors": neighbors,
        "threshold": float(np.percentile(nearest_other_test[:, 0], support["range_percentile"])),
        "reference_rows": np.asarray(reference_rows),
    }


def fit_support(features: pd.DataFrame, support: dict[str, Any]) -> dict[str, Any]:
    columns = list(support["columns"])
    partitions = {}
    for value, members in features.groupby(support["partition_column"], sort=True):
        partitions[str(value)] = fit_partition(members[columns].to_numpy(dtype=float), members.index.to_numpy(),
                                               support)
    return {
        "columns": columns,
        "partition_column": support["partition_column"],
        "range_percentile": support["range_percentile"],
        "nearest_k": support["nearest_k"],
        "partitions": partitions,
    }


def support_thresholds(bundle: dict[str, Any]) -> dict[str, float]:
    return {name: partition["threshold"] for name, partition in bundle["partitions"].items()}


def query_support(bundle: dict[str, Any], feature_row: pd.DataFrame) -> dict[str, Any]:
    partition_value = str(feature_row[bundle["partition_column"]].iloc[0])
    if partition_value not in bundle["partitions"]:
        raise ValueError(f"No support data for {bundle['partition_column']}={partition_value!r}")
    partition = bundle["partitions"][partition_value]
    scaled = partition["scaler"].transform(feature_row[bundle["columns"]].to_numpy(dtype=float))
    neighbors = partition["neighbors"]
    distances, positions = neighbors.kneighbors(scaled, n_neighbors=min(bundle["nearest_k"],
                                                                        neighbors.n_samples_fit_))
    within_threshold = neighbors.radius_neighbors(scaled, radius=partition["threshold"], return_distance=False)[0]
    return {
        "within_joint_range": bool(distances[0, 0] <= partition["threshold"]),
        "support_n": int(len(within_threshold)),
        "nearest_reference_rows": [int(row) for row in partition["reference_rows"][positions[0]]],
        "nearest_distances": [float(distance) for distance in distances[0]],
        "threshold": partition["threshold"],
    }


def confidence_level(in_tested_range: bool, support_n: int, top_probability: float | None,
                     rule: dict[str, Any]) -> str:
    if not in_tested_range or support_n < rule["low_confidence_below_support"]:
        return "low"
    enough_support = support_n >= rule["high_confidence_min_support"]
    min_probability = rule.get("high_confidence_min_probability")
    confident_prediction = (min_probability is None or
                            (top_probability is not None and top_probability >= min_probability))
    return "high" if enough_support and confident_prediction else "medium"
