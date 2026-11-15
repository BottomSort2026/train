from __future__ import annotations

import copy
import hashlib

import pytest

from flame.training.config import load_config, validate_config


def test_outcome_config_has_one_baseline_and_two_candidates(outcome_config):
    assert outcome_config.role == "publishable"
    assert outcome_config.task == "classification"
    assert outcome_config.model_version == "flex-outcome-0.1"
    assert outcome_config.baseline_name == "logreg"
    assert outcome_config.candidate_names == ["logreg_spline", "hgb"]


def test_outcome_config_keeps_the_decided_cross_validation():
    cv = load_config("configs/training/flex_outcome.toml").section("cv")
    assert (cv["n_splits"], cv["n_repeats"], cv["shuffle"]) == (5, 10, True)


def test_feature_columns_follow_the_decided_pipeline_inputs(outcome_config):
    assert outcome_config.feature_columns == ["fuel", "pressure_atm", "o2", "co2", "he", "d0_mm", "fiber"]


def test_sensitivity_config_inherits_models_and_overrides_the_target(sensitivity_config, outcome_config):
    assert sensitivity_config.role == "sensitivity"
    assert sensitivity_config.reference is not None
    assert sensitivity_config.section("models") == outcome_config.section("models")
    assert sensitivity_config.section("target")["class_map"] == {"extinction": "extinction",
                                                                 "completion": "no_extinction"}
    assert sensitivity_config.section("target")["positive_class"] == "extinction"
    assert [item["name"] for item in sensitivity_config.section("dataset")["extra_exclusions"]] == ["disruption"]


def test_config_without_a_baseline_is_rejected(outcome_config):
    values = copy.deepcopy(outcome_config.values)
    values["models"]["logreg"]["role"] = "candidate"
    with pytest.raises(ValueError, match="baseline"):
        validate_config(values)


def test_classification_config_needs_two_classes(outcome_config):
    values = copy.deepcopy(outcome_config.values)
    values["target"]["class_map"] = {"extinction": "extinction"}
    with pytest.raises(ValueError, match="two classes"):
        validate_config(values)


def test_config_missing_a_section_is_rejected(outcome_config):
    values = copy.deepcopy(outcome_config.values)
    del values["support"]
    with pytest.raises(ValueError, match="support"):
        validate_config(values)


def test_calibrated_model_needs_calibration_settings(outcome_config):
    values = copy.deepcopy(outcome_config.values)
    del values["calibration"]
    with pytest.raises(ValueError, match="calibration"):
        validate_config(values)


def test_config_hash_and_values_ignore_line_endings(outcome_config, tmp_path):
    content = outcome_config.path.read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "lf.toml").write_bytes(content)
    (tmp_path / "crlf.toml").write_bytes(content.replace(b"\n", b"\r\n"))
    lf, crlf = load_config(tmp_path / "lf.toml"), load_config(tmp_path / "crlf.toml")
    assert crlf.sha256 == lf.sha256 == outcome_config.sha256 == hashlib.sha256(content).hexdigest()
    assert crlf.values == lf.values
