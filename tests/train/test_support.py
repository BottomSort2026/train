from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from flame.training.support import confidence_level, fit_support, query_support

RULE = {"low_confidence_below_support": 3, "high_confidence_min_support": 5, "high_confidence_min_probability": 0.7}


def standardized_partition(features: pd.DataFrame, bundle: dict, fuel: str) -> np.ndarray:
    members = features[features["fuel"] == fuel]
    return bundle["partitions"][fuel]["scaler"].transform(members[bundle["columns"]].to_numpy(dtype=float))


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(axis=2))


@pytest.fixture(scope="module")
def support_bundle(synthetic_data, outcome_config):
    return fit_support(synthetic_data.features, outcome_config.support)


def test_one_index_is_built_per_fuel_over_the_decided_columns(support_bundle):
    assert sorted(support_bundle["partitions"]) == ["heptane", "methanol"]
    assert support_bundle["columns"] == ["pressure_atm", "o2", "co2", "he", "d0_mm"]
    assert "fiber" not in support_bundle["columns"]


def test_threshold_is_the_percentile_of_distances_to_the_nearest_other_test(synthetic_data, support_bundle,
                                                                           outcome_config):
    percentile = outcome_config.support.range_percentile
    for fuel, partition in support_bundle["partitions"].items():
        distances = pairwise_distances(standardized_partition(synthetic_data.features, support_bundle, fuel))
        np.fill_diagonal(distances, np.inf)
        assert partition["threshold"] == pytest.approx(np.percentile(distances.min(axis=1), percentile))


def test_support_n_counts_same_fuel_training_rows_within_the_threshold(synthetic_data, support_bundle):
    query = synthetic_data.features.iloc[[20]]
    fuel = query["fuel"].iloc[0]
    partition = support_bundle["partitions"][fuel]
    points = standardized_partition(synthetic_data.features, support_bundle, fuel)
    scaled_query = partition["scaler"].transform(query[support_bundle["columns"]].to_numpy(dtype=float))
    expected = int((np.sqrt(((points - scaled_query) ** 2).sum(axis=1)) <= partition["threshold"]).sum())
    result = query_support(support_bundle, query)
    assert result["support_n"] == expected
    assert result["within_joint_range"] is True


def test_nearest_rows_are_same_fuel_training_rows_sorted_by_distance(synthetic_data, support_bundle):
    query = synthetic_data.features.iloc[[20]]
    result = query_support(support_bundle, query)
    nearest = synthetic_data.features.iloc[result["nearest_reference_rows"]]
    assert result["nearest_reference_rows"][0] == 20
    assert result["nearest_distances"][0] == pytest.approx(0.0)
    assert result["nearest_distances"] == sorted(result["nearest_distances"])
    assert len(result["nearest_reference_rows"]) == support_bundle["nearest_k"]
    assert (nearest["fuel"] == query["fuel"].iloc[0]).all()


def test_query_far_from_every_test_is_outside_the_joint_range(synthetic_data, support_bundle):
    query = synthetic_data.features.iloc[[20]].copy()
    query["o2"] = 0.9
    result = query_support(support_bundle, query)
    assert result["within_joint_range"] is False
    assert result["support_n"] == 0


def test_untested_combination_inside_every_single_range_is_outside_the_joint_range(outcome_config):
    rng = np.random.default_rng(11)
    jitter = rng.normal(0, 0.002, size=(40, 2))
    high_oxygen_no_suppressant = np.column_stack([0.34 + jitter[:20, 0], np.zeros(20)])
    low_oxygen_with_co2 = np.column_stack([0.18 + jitter[20:, 0], 0.30 + jitter[20:, 1]])
    mixture = np.vstack([high_oxygen_no_suppressant, low_oxygen_with_co2])
    features = pd.DataFrame({"fuel": "methanol", "pressure_atm": 0.7 + rng.normal(0, 0.002, 40), "o2": mixture[:, 0],
                             "co2": mixture[:, 1], "he": 0.0, "d0_mm": 3.0 + rng.normal(0, 0.05, 40)})
    bundle = fit_support(features, outcome_config.support)
    tested = features.iloc[[0]]
    untested = tested.assign(o2=0.34, co2=0.30)
    assert features["o2"].min() <= 0.34 <= features["o2"].max() + 0.01
    assert features["co2"].min() <= 0.30 <= features["co2"].max() + 0.01
    assert query_support(bundle, tested)["within_joint_range"] is True
    assert query_support(bundle, untested)["within_joint_range"] is False
    assert query_support(bundle, untested)["support_n"] == 0


def test_unknown_fuel_has_no_support_index(synthetic_data, support_bundle):
    query = synthetic_data.features.iloc[[20]].copy()
    query["fuel"] = "ethanol"
    with pytest.raises(ValueError, match="ethanol"):
        query_support(support_bundle, query)


@pytest.mark.parametrize(("in_range", "support_n", "probability", "expected"), [
    (False, 50, 0.99, "low"),
    (True, 2, 0.99, "low"),
    (True, 3, 0.99, "medium"),
    (True, 4, 0.99, "medium"),
    (True, 5, 0.69, "medium"),
    (True, 5, 0.70, "high"),
    (True, 50, 0.99, "high"),
])
def test_confidence_boundaries(in_range, support_n, probability, expected):
    assert confidence_level(in_range, support_n, probability, RULE) == expected


def test_confidence_without_a_probability_rule_uses_support_only():
    rule = {"low_confidence_below_support": 3, "high_confidence_min_support": 5}
    assert confidence_level(True, 5, None, rule) == "high"
    assert confidence_level(True, 4, None, rule) == "medium"
    assert confidence_level(False, 9, None, rule) == "low"
