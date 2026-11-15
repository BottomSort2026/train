from __future__ import annotations

import copy
import dataclasses
import hashlib

import pytest

from configs.training.config_training import (
    MetricRule,
    experiment_from_values,
    get_experiment,
    init_training_settings,
    load_experiment,
    load_values,
    training_settings,
)

from conftest import EXTINCTION_CONFIG, OUTCOME_CONFIG, SENSITIVITY_CONFIG


def build(values):
    return experiment_from_values(values, path=load_experiment(OUTCOME_CONFIG).path, sha256="0" * 64)


@pytest.fixture()
def outcome_values():
    values, _, _ = load_values(OUTCOME_CONFIG)
    return copy.deepcopy(values)


def test_every_yaml_is_loaded_once_at_import_keyed_by_experiment_name():
    assert sorted(training_settings) == ["flex-extinction", "flex-outcome", "flex-outcome-sensitivity"]
    assert init_training_settings() == training_settings


def test_experiments_are_found_by_name_or_by_path():
    assert get_experiment("flex-outcome") is training_settings["flex-outcome"]
    assert get_experiment(OUTCOME_CONFIG) is training_settings["flex-outcome"]
    with pytest.raises(ValueError, match="known experiments"):
        get_experiment("flex-unknown")
    with pytest.raises(ValueError, match="YAML"):
        get_experiment("configs/training/flex_outcome.toml")


def test_settings_are_frozen(outcome_config):
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome_config.cv.n_splits = 3


def test_outcome_config_has_one_baseline_and_three_candidates(outcome_config):
    assert outcome_config.role == "publishable"
    assert outcome_config.task == "classification"
    assert outcome_config.model_version == "flex-outcome-0.2"
    assert outcome_config.baseline_name == "logreg"
    assert outcome_config.candidate_names == ["logreg_spline", "hgb", "logreg_calibrated"]
    assert outcome_config.models["logreg_calibrated"].calibrated
    assert outcome_config.models["logreg_calibrated"].params == outcome_config.models["logreg"].params


def test_outcome_config_keeps_the_decided_cross_validation():
    cv = load_experiment(OUTCOME_CONFIG).cv
    assert (cv.n_splits, cv.n_repeats, cv.shuffle) == (5, 10, True)


def test_feature_columns_follow_the_decided_pipeline_inputs(outcome_config):
    assert outcome_config.feature_columns == ["fuel", "pressure_atm", "o2", "co2", "he", "d0_mm", "fiber"]


def test_sensitivity_config_inherits_models_and_overrides_the_target(sensitivity_config, outcome_config):
    assert sensitivity_config.role == "sensitivity"
    assert sensitivity_config.reference is not None
    assert sensitivity_config.models == outcome_config.models
    assert sensitivity_config.target.class_map == {"extinction": "extinction", "completion": "no_extinction"}
    assert sensitivity_config.target.positive_class == "extinction"
    assert [item.name for item in sensitivity_config.dataset.extra_exclusions] == ["disruption"]
    assert sensitivity_config.comparison.reference_config == outcome_config.path


def test_extinction_config_has_no_probability_rule():
    support = load_experiment(EXTINCTION_CONFIG).support
    assert support.high_confidence_min_probability is None
    assert "high_confidence_min_probability" not in support.confidence_rule()


def test_config_without_a_baseline_is_rejected(outcome_values):
    outcome_values["models"]["logreg"]["role"] = "candidate"
    with pytest.raises(ValueError, match="baseline"):
        build(outcome_values)


def test_classification_config_needs_two_classes(outcome_values):
    outcome_values["target"]["class_map"] = {"extinction": "extinction"}
    with pytest.raises(ValueError, match="two classes"):
        build(outcome_values)


def test_config_missing_a_section_is_rejected(outcome_values):
    del outcome_values["support"]
    with pytest.raises(ValueError, match="support"):
        build(outcome_values)


def test_calibrated_model_needs_calibration_settings(outcome_values):
    del outcome_values["calibration"]
    with pytest.raises(ValueError, match="calibration"):
        build(outcome_values)


@pytest.mark.parametrize(("section", "key"), [("cv", "n_split"), ("support", "nearestk"), ("experiment", "taks")])
def test_unknown_keys_are_rejected(outcome_values, section, key):
    outcome_values[section][key] = 1
    with pytest.raises(ValueError, match="Unknown keys"):
        build(outcome_values)


def test_unknown_estimator_is_rejected(outcome_values):
    outcome_values["models"]["hgb"]["estimator"] = "random_forest"
    with pytest.raises(ValueError, match="estimator"):
        build(outcome_values)


def test_regression_estimator_in_a_classification_config_is_rejected(outcome_values):
    outcome_values["models"]["hgb"] = {"role": "candidate", "estimator": "ridge", "params": {"alpha": 1.0}}
    with pytest.raises(ValueError, match="classification"):
        build(outcome_values)


def test_exclusion_rule_that_compares_needs_a_value(outcome_values):
    outcome_values["dataset"]["exclusions"].append({"name": "only_heptane", "column": "fuel", "rule": "equals"})
    with pytest.raises(ValueError, match="needs a value"):
        build(outcome_values)


def test_unquoted_version_is_rejected(outcome_values):
    outcome_values["experiment"]["version"] = 0.2
    with pytest.raises(ValueError, match="quoted string"):
        build(outcome_values)


def test_input_order_must_list_every_exposed_input(outcome_values):
    outcome_values["interface"]["input_order"].remove("droplet_d0_mm")
    with pytest.raises(ValueError, match="input_order"):
        build(outcome_values)


def test_selection_rules_are_typed(outcome_config):
    assert outcome_config.selection.must_beat == (MetricRule("macro_f1", "higher"), MetricRule("brier", "lower"))


def test_config_hash_and_values_ignore_line_endings(outcome_config, tmp_path):
    content = outcome_config.path.read_bytes().replace(b"\r\n", b"\n")
    (tmp_path / "lf.yml").write_bytes(content)
    (tmp_path / "crlf.yml").write_bytes(content.replace(b"\n", b"\r\n"))
    lf, crlf = load_experiment(tmp_path / "lf.yml"), load_experiment(tmp_path / "crlf.yml")
    assert crlf.sha256 == lf.sha256 == outcome_config.sha256 == hashlib.sha256(content).hexdigest()
    assert dataclasses.replace(crlf, path=lf.path) == lf


def test_sensitivity_file_hash_is_its_own(sensitivity_config, outcome_config):
    assert sensitivity_config.sha256 != outcome_config.sha256
    assert sensitivity_config.path.name == SENSITIVITY_CONFIG.rsplit("/", 1)[-1]
