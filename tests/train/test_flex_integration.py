from __future__ import annotations

import json

import pytest

from configs.training.config_training import load_experiment as load_config
from flame.training import flex_check, flex_data
from flame.training.datasets import load_training_data
from flame.training.evaluate import evaluation_splits
from flame.training.export import ARTIFACTS_ROOT, export_experiment
from flame.training.inference import load_bundle, reference_predict

from conftest import EXTINCTION_CONFIG, OUTCOME_CONFIG, SENSITIVITY_CONFIG, faster

pytestmark = pytest.mark.skipif(not flex_data.PROCESSED_CSV.exists(),
                                reason="data/processed/flex_tests.csv is missing; run prepare-flex")

MOON_WITH_CO2 = {"fuel": "methanol", "pressure_atm": 0.7, "o2": 0.34, "suppressant": "co2",
                 "suppressant_fraction": 0.2, "droplet_d0_mm": 3.5}
ISS_AIR = {"fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "suppressant": "none", "suppressant_fraction": 0.0,
           "droplet_d0_mm": 3.0}
MOON_NO_SUPPRESSANT = {"fuel": "methanol", "pressure_atm": 0.69, "o2": 0.34, "suppressant": "none",
                       "suppressant_fraction": 0.0, "droplet_d0_mm": 3.5}


@pytest.fixture(scope="module")
def flex_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("flex_export")
    result = export_experiment(faster(load_config(OUTCOME_CONFIG)), root / "artifacts", root / "reports")
    return load_bundle(result["artifact_dir"])


def test_training_rows_match_decision_d03():
    data = load_training_data(load_config(OUTCOME_CONFIG))
    assert len(data.rows) == 251
    assert data.target.value_counts().to_dict() == {"extinction": 171, "no_extinction": 80}
    assert data.exclusions == {"duplicate_test": 1, "missing_initial_diameter": 12, "high_pressure": 9,
                               "unknown_fiber": 1}


def test_sensitivity_rows_match_decision_d02():
    data = load_training_data(load_config(SENSITIVITY_CONFIG))
    assert len(data.rows) == 198
    assert data.target.value_counts().to_dict() == {"extinction": 171, "no_extinction": 27}


def test_extinction_rows_match_decision_d08():
    data = load_training_data(load_config(EXTINCTION_CONFIG))
    assert len(data.rows) == 156
    assert set(data.rows["test_end"]) == {"extinction"}
    assert data.rows["dext_mm"].notna().all()
    assert data.exclusions["not_extinction"] == 80


def test_co2_and_helium_are_never_mixed():
    table = flex_data.load_flex()
    assert int(((table["co2"] > 0) & (table["he"] > 0)).sum()) == 0


def test_real_folds_never_share_an_atmosphere():
    config = load_config(OUTCOME_CONFIG)
    data = load_training_data(config)
    groups = data.groups.to_numpy()
    for split in evaluation_splits(data, config):
        assert not set(groups[split.train]) & set(groups[split.test])


def test_tested_ranges_come_from_the_flex_rows(flex_bundle):
    inputs = flex_bundle["info"]["inputs"]
    assert (inputs["pressure_atm"]["min"], inputs["pressure_atm"]["max"]) == (0.69, 1.02)
    assert (inputs["o2"]["min"], inputs["o2"]["max"]) == (0.12, 0.34)
    assert inputs["suppressant"]["values"] == ["none", "co2", "he"]


def test_moon_like_atmosphere_with_co2_has_no_supporting_tests(flex_bundle):
    response = reference_predict(flex_bundle, MOON_WITH_CO2)
    assert response["out_of_range_fields"] == []
    assert response["in_tested_range"] is False
    assert response["confidence"] == "low"


def test_iss_like_air_is_in_range_with_enough_support(flex_bundle):
    response = reference_predict(flex_bundle, ISS_AIR)
    assert response["in_tested_range"] is True
    assert response["support_n"] >= 5


def test_moon_like_atmosphere_without_suppressant_is_in_range(flex_bundle):
    response = reference_predict(flex_bundle, MOON_NO_SUPPRESSANT)
    assert response["in_tested_range"] is True
    assert response["out_of_range_fields"] == []
    assert response["outcome"] == "no_extinction"


def test_published_artifact_was_trained_on_the_current_data():
    info_path = ARTIFACTS_ROOT / "flex-outcome-0.2" / "model_info.json"
    if not info_path.exists():
        pytest.skip("artifacts/models/flex-outcome-0.2 has not been exported")
    manifest = json.loads(flex_data.MANIFEST.read_text(encoding="utf-8"))
    assert json.loads(info_path.read_text(encoding="utf-8"))["training_source"]["sha256"] == manifest["sha256"]


def test_data_check_report_is_utf8_with_unix_line_endings(tmp_path):
    text = flex_check.run_check(tmp_path / "flex_data_check.md").read_bytes().decode("utf-8")
    assert "\r" not in text
    assert "≈0.7 atm" in text
