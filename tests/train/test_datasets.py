from __future__ import annotations

import pandas as pd
import pytest

from flame.training.datasets import apply_exclusions, build_training_data, exclusion_mask

from conftest import EXCLUDED_BY_FILTER, SYNTHETIC_SHA256


def test_exclusions_are_counted_per_filter_in_order(synthetic_table, synthetic_data):
    assert synthetic_data.exclusions == EXCLUDED_BY_FILTER
    assert len(synthetic_data.rows) == len(synthetic_table) - sum(EXCLUDED_BY_FILTER.values())
    assert synthetic_data.source_rows == len(synthetic_table)


def test_a_row_matching_two_filters_is_counted_once():
    table = pd.DataFrame({"a": [1.0, None, None], "b": [True, True, False]})
    kept, removed = apply_exclusions(table, [{"name": "missing_a", "column": "a", "rule": "is_null"},
                                             {"name": "flagged_b", "column": "b", "rule": "is_true"}])
    assert removed == {"missing_a": 2, "flagged_b": 1}
    assert len(kept) == 0


def test_unknown_exclusion_rule_is_rejected():
    with pytest.raises(ValueError, match="Unknown exclusion rule"):
        exclusion_mask(pd.Series([1, 2]), "greater_than", 1)


def test_classification_target_joins_disruption_and_completion(synthetic_data):
    joined = synthetic_data.rows["test_end"].isin(["disruption", "completion"])
    assert set(synthetic_data.target.unique()) == {"extinction", "no_extinction"}
    assert (synthetic_data.target[joined] == "no_extinction").all()
    assert (synthetic_data.target[~joined] == "extinction").all()


def test_target_value_without_a_class_is_rejected(synthetic_table, outcome_config):
    table = synthetic_table.copy()
    table.loc[10, "test_end"] = "no_ignition"
    with pytest.raises(ValueError, match="no_ignition"):
        build_training_data(table, outcome_config, SYNTHETIC_SHA256)


def test_feature_frame_has_the_configured_columns_and_types(synthetic_data, outcome_config):
    features = synthetic_data.features
    assert list(features.columns) == outcome_config.feature_columns
    assert features["fiber"].dtype == bool
    assert features["pressure_atm"].dtype == float
    assert not features.isna().any().any()


def test_features_target_groups_and_rows_stay_aligned(synthetic_data):
    assert len(synthetic_data.features) == len(synthetic_data.target) == len(synthetic_data.groups)
    assert list(synthetic_data.features.index) == list(range(len(synthetic_data.rows)))
    assert (synthetic_data.groups == synthetic_data.rows["atmosphere_id"]).all()
    assert synthetic_data.row_keys.is_unique


def test_sensitivity_rows_are_the_reference_rows_without_disruption(synthetic_table, synthetic_data,
                                                                    sensitivity_config):
    sensitivity = build_training_data(synthetic_table, sensitivity_config, SYNTHETIC_SHA256)
    expected = synthetic_data.rows[synthetic_data.rows["test_end"] != "disruption"]
    assert sensitivity.exclusions["disruption"] == int((synthetic_data.rows["test_end"] == "disruption").sum())
    assert list(sensitivity.row_keys) == list(expected["table_row"])
    assert "disruption" not in set(sensitivity.rows["test_end"])
