from __future__ import annotations

import numpy as np
import pytest

from flame.training.datasets import build_training_data
from flame.training.sensitivity import compare_targets_from_data, reference_positions, spearman_correlation

from conftest import SYNTHETIC_SHA256


@pytest.fixture(scope="module")
def sensitivity_data(synthetic_table, sensitivity_config):
    return build_training_data(synthetic_table, sensitivity_config, SYNTHETIC_SHA256)


def test_shared_rows_map_back_to_the_same_tests(synthetic_data, sensitivity_data):
    positions = reference_positions(synthetic_data, sensitivity_data)
    assert len(positions) == len(sensitivity_data.rows)
    assert list(synthetic_data.row_keys.to_numpy()[positions]) == list(sensitivity_data.row_keys)


def test_sensitivity_rows_outside_the_reference_are_rejected(synthetic_data, sensitivity_data):
    with pytest.raises(ValueError, match="subset"):
        reference_positions(sensitivity_data, synthetic_data)


def test_spearman_measures_agreement_of_rankings():
    values = np.array([0.1, 0.4, 0.2, 0.9])
    assert spearman_correlation(values, values ** 3) == pytest.approx(1.0)
    assert spearman_correlation(values, -values) == pytest.approx(-1.0)


def test_comparison_reports_shared_rows_and_a_mechanical_verdict(synthetic_data, sensitivity_data,
                                                                 sensitivity_config):
    result = compare_targets_from_data(sensitivity_config, synthetic_data, sensitivity_data)
    comparison = sensitivity_config.section("comparison")
    assert result["shared_rows"] == len(sensitivity_data.rows) < result["reference_rows"]
    assert "disruption" not in result["sensitivity_counts"]
    assert 0.0 <= result["reference_auc"] <= 1.0 and 0.0 <= result["sensitivity_auc"] <= 1.0
    assert result["auc_delta"] == pytest.approx(result["reference_auc"] - result["sensitivity_auc"])
    expected = (abs(result["auc_delta"]) <= comparison["max_abs_roc_auc_delta"]
                and result["spearman"] >= comparison["min_spearman"])
    assert result["consistent"] is expected
