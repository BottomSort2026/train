from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from flame.training import inference
from flame.training.inference import build_feature_row, reference_predict, validate_inputs

RESPONSE_FIELDS = ["outcome", "probabilities", "in_tested_range", "out_of_range_fields", "support_n", "nearest_tests",
                   "confidence", "model_version", "training_source"]


def test_response_has_the_contract_fields_and_types(synthetic_export, typical_inputs):
    response = reference_predict(synthetic_export["bundle"], typical_inputs)
    assert list(response) == RESPONSE_FIELDS
    assert response["outcome"] in ("extinction", "no_extinction")
    assert isinstance(response["probabilities"], dict)
    assert all(isinstance(value, float) for value in response["probabilities"].values())
    assert isinstance(response["in_tested_range"], bool)
    assert isinstance(response["out_of_range_fields"], list)
    assert isinstance(response["support_n"], int)
    assert isinstance(response["nearest_tests"], list)
    assert response["confidence"] in ("low", "medium", "high")
    assert response["model_version"] == "flex-outcome-0.2"
    assert sorted(response["training_source"]) == ["accession", "dataset", "rows", "sha256"]
    assert json.loads(json.dumps(response)) == response


def test_probabilities_cover_both_classes_and_decide_the_outcome(synthetic_export, typical_inputs):
    info = synthetic_export["bundle"]["info"]
    response = reference_predict(synthetic_export["bundle"], typical_inputs)
    probabilities = response["probabilities"]
    assert sorted(probabilities) == ["extinction", "no_extinction"]
    assert sum(probabilities.values()) == pytest.approx(1.0)
    expected = "extinction" if probabilities["extinction"] >= info["decision_threshold"] else "no_extinction"
    assert response["outcome"] == expected


def test_nearest_tests_carry_ids_and_increasing_distances(synthetic_export, typical_inputs):
    tests = reference_predict(synthetic_export["bundle"], typical_inputs)["nearest_tests"]
    assert len(tests) == synthetic_export["bundle"]["info"]["support"]["nearest_k"]
    assert all(list(test) == ["cir_id", "testsheet_id", "test_end", "conditions", "distance"] for test in tests)
    assert [test["distance"] for test in tests] == sorted(test["distance"] for test in tests)


def test_nearest_tests_carry_the_conditions_of_the_real_test(synthetic_export, synthetic_data, typical_inputs):
    tests = reference_predict(synthetic_export["bundle"], typical_inputs)["nearest_tests"]
    rows = synthetic_data.rows.set_index("cir_id")
    for test in tests:
        conditions = test["conditions"]
        assert list(conditions) == ["fuel", "pressure_atm", "o2", "co2", "he", "d0_mm"]
        assert conditions["fuel"] == typical_inputs["fuel"]
        assert all(isinstance(conditions[name], float) for name in ("pressure_atm", "o2", "co2", "he", "d0_mm"))
        assert conditions["o2"] == pytest.approx(float(rows.loc[test["cir_id"], "o2"]))
        assert conditions["d0_mm"] == pytest.approx(float(rows.loc[test["cir_id"], "d0_mm"]))
    assert json.loads(json.dumps(tests)) == tests


@pytest.mark.parametrize(("suppressant", "fraction", "expected_co2", "expected_he"), [
    ("co2", 0.2, 0.2, 0.0),
    ("he", 0.3, 0.0, 0.3),
    ("none", 0.0, 0.0, 0.0),
])
def test_suppressant_choice_maps_to_the_co2_and_he_columns(synthetic_export, typical_inputs, suppressant, fraction,
                                                           expected_co2, expected_he):
    info = synthetic_export["bundle"]["info"]
    inputs = {**typical_inputs, "suppressant": suppressant, "suppressant_fraction": fraction}
    row = build_feature_row(info, inputs)
    assert list(row.columns) == ["fuel", "pressure_atm", "o2", "co2", "he", "d0_mm", "fiber"]
    assert row["co2"].iloc[0] == pytest.approx(expected_co2)
    assert row["he"].iloc[0] == pytest.approx(expected_he)
    assert row["d0_mm"].iloc[0] == pytest.approx(typical_inputs["droplet_d0_mm"])


def test_fiber_is_fixed_to_a_free_droplet_and_never_exposed(synthetic_export, typical_inputs):
    info = synthetic_export["bundle"]["info"]
    assert "fiber" not in info["inputs"]
    assert info["fixed_inputs"] == {"fiber": False}
    assert bool(build_feature_row(info, typical_inputs)["fiber"].iloc[0]) is False
    with pytest.raises(ValueError, match="unexpected"):
        reference_predict(synthetic_export["bundle"], {**typical_inputs, "fiber": True})


def test_exposed_inputs_are_the_decided_interface(synthetic_export):
    assert list(synthetic_export["bundle"]["info"]["inputs"]) == [
        "fuel", "pressure_atm", "o2", "suppressant", "suppressant_fraction", "droplet_d0_mm"]


def test_missing_input_is_rejected(synthetic_export, typical_inputs):
    inputs = {key: value for key, value in typical_inputs.items() if key != "o2"}
    with pytest.raises(ValueError, match="missing"):
        reference_predict(synthetic_export["bundle"], inputs)


def test_unknown_fuel_is_rejected(synthetic_export, typical_inputs):
    with pytest.raises(ValueError, match="fuel"):
        reference_predict(synthetic_export["bundle"], {**typical_inputs, "fuel": "ethanol"})


def test_non_numeric_value_is_rejected(synthetic_export, typical_inputs):
    with pytest.raises(ValueError, match="finite number"):
        reference_predict(synthetic_export["bundle"], {**typical_inputs, "o2": "high"})


def test_fraction_without_a_suppressant_is_rejected(synthetic_export, typical_inputs):
    inputs = {**typical_inputs, "suppressant": "none", "suppressant_fraction": 0.1}
    with pytest.raises(ValueError, match="must be 0"):
        reference_predict(synthetic_export["bundle"], inputs)


def test_mixture_above_one_is_rejected(synthetic_export, typical_inputs):
    inputs = {**typical_inputs, "o2": 0.9, "suppressant_fraction": 0.2}
    with pytest.raises(ValueError, match="must be <= 1"):
        reference_predict(synthetic_export["bundle"], inputs)


def test_value_outside_its_tested_range_is_reported_with_low_confidence(synthetic_export, typical_inputs):
    info = synthetic_export["bundle"]["info"]
    inputs = {**typical_inputs, "o2": info["inputs"]["o2"]["max"] + 0.1}
    assert validate_inputs(info, inputs) == ["o2"]
    response = reference_predict(synthetic_export["bundle"], inputs)
    assert response["out_of_range_fields"] == ["o2"]
    assert response["in_tested_range"] is False
    assert response["confidence"] == "low"


def test_suppressant_fraction_is_checked_against_the_range_of_its_own_suppressant(synthetic_export, typical_inputs):
    info = synthetic_export["bundle"]["info"]
    helium_max = info["inputs"]["suppressant_fraction"]["by_selector"]["he"]["max"]
    co2_max = info["inputs"]["suppressant_fraction"]["by_selector"]["co2"]["max"]
    assert helium_max > co2_max
    inputs = {**typical_inputs, "suppressant": "co2", "suppressant_fraction": helium_max}
    assert validate_inputs(info, inputs) == ["suppressant_fraction"]
    assert validate_inputs(info, {**inputs, "suppressant": "he"}) == []


def test_module_is_self_contained_so_engine_can_copy_it_verbatim():
    allowed = {"__future__", "json", "math", "pathlib", "typing", "joblib", "numpy", "pandas"}
    tree = ast.parse(Path(inference.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= allowed, f"engine cannot import {sorted(imported - allowed)}"
