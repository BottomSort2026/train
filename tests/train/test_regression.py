from __future__ import annotations

import numpy as np
import pytest

from flame.training.config import load_config, validate_config
from flame.training.datasets import build_training_data
from flame.training.evaluate import absolute_error_by_region, evaluate_models, evaluation_splits
from flame.training.export import export_from_data
from flame.training.inference import load_bundle, reference_predict
from flame.training.models import fit_model, predict_values
from flame.training.splits import stratification_labels

from conftest import EXTINCTION_CONFIG, SYNTHETIC_SHA256, faster

RESPONSE_FIELDS = ["extinction_diameter_ratio", "extinction_diameter_mm", "in_tested_range", "out_of_range_fields",
                   "support_n", "nearest_tests", "confidence", "model_version", "training_source"]


@pytest.fixture(scope="module")
def extinction_config():
    return faster(load_config(EXTINCTION_CONFIG))


@pytest.fixture(scope="module")
def extinction_data(synthetic_table, extinction_config):
    return build_training_data(synthetic_table, extinction_config, SYNTHETIC_SHA256)


@pytest.fixture(scope="module")
def extinction_export(synthetic_table, extinction_data, extinction_config, tmp_path_factory):
    root = tmp_path_factory.mktemp("extinction_export")
    result = export_from_data(extinction_config, extinction_data, synthetic_table, root / "artifacts",
                              root / "reports")
    return {**result, "bundle": load_bundle(result["artifact_dir"])}


def test_extinction_config_is_a_regression_with_a_ridge_baseline(extinction_config):
    assert extinction_config.task == "regression"
    assert extinction_config.model_version == "flex-extinction-0.2"
    assert extinction_config.baseline_name == "ridge"
    assert extinction_config.candidate_names == ["hgb"]
    assert extinction_config.section("selection")["must_beat"] == [{"metric": "mae", "direction": "lower"}]


def test_regression_config_needs_an_output_name_and_strata(extinction_config):
    values = {name: dict(section) for name, section in extinction_config.values.items()}
    del values["cv"]["stratify_bins"]
    with pytest.raises(ValueError, match="stratify_bins"):
        validate_config(values)


def test_only_extinction_rows_with_a_measured_diameter_are_used(synthetic_table, extinction_data):
    usable = synthetic_table[(synthetic_table["test_end"] == "extinction") & synthetic_table["dext_mm"].notna()
                             & synthetic_table["d0_mm"].notna() & ~synthetic_table["high_pressure"]
                             & synthetic_table["duplicate_of"].isna() & synthetic_table["fiber"].notna()]
    assert set(extinction_data.rows["test_end"]) == {"extinction"}
    assert list(extinction_data.row_keys) == list(usable["table_row"])
    assert extinction_data.target.between(0, 1).all()
    assert extinction_data.target.equals(extinction_data.rows["dext_over_d0"].rename("target"))


def test_regression_folds_are_grouped_and_stratified_on_target_bins(extinction_data, extinction_config):
    labels = stratification_labels(extinction_data.target, extinction_config)
    assert labels.nunique() == extinction_config.section("cv")["stratify_bins"]
    groups = extinction_data.groups.to_numpy()
    for split in evaluation_splits(extinction_data, extinction_config):
        assert not set(groups[split.train]) & set(groups[split.test])


def test_regression_is_scored_with_error_metrics_for_every_model(extinction_data, extinction_config):
    evaluation = evaluate_models(extinction_data, extinction_config)
    assert evaluation.metric_names == ["mae", "rmse", "r2"]
    assert set(evaluation.fold_metrics["model"]) == {"ridge", "hgb"}
    assert evaluation.selection["baseline"] == "ridge"
    assert evaluation.selection["selected"] in ("ridge", "hgb")
    regions = absolute_error_by_region(evaluation, extinction_data, extinction_config)
    assert list(regions.columns) == ["rows", "ridge", "hgb"]
    assert int(regions["rows"].sum()) == len(extinction_data.rows)
    assert all("pressure_atm=" in region and "suppressant=" in region for region in regions.index)


def test_predictions_are_clipped_to_the_physical_bounds(extinction_data, extinction_config):
    pipeline = fit_model("ridge", extinction_config, extinction_data.features, extinction_data.target,
                         extinction_data.groups, seed=0)
    extreme = extinction_data.features.iloc[:2].copy()
    extreme["o2"] = [0.01, 0.99]
    raw = pipeline.predict(extreme)
    clipped = predict_values(pipeline, extreme, extinction_config)
    assert raw.max() > 1.0 or raw.min() < 0.0
    assert np.all((clipped >= 0.0) & (clipped <= 1.0))


def test_regression_response_gives_the_ratio_and_the_diameter(extinction_export, typical_inputs):
    response = reference_predict(extinction_export["bundle"], typical_inputs)
    assert list(response) == RESPONSE_FIELDS
    assert 0.0 <= response["extinction_diameter_ratio"] <= 1.0
    assert response["extinction_diameter_mm"] == pytest.approx(
        response["extinction_diameter_ratio"] * typical_inputs["droplet_d0_mm"])
    assert response["confidence"] in ("low", "medium", "high")
    assert response["model_version"] == "flex-extinction-0.2"


def test_regression_model_info_describes_the_outputs_and_bounds(extinction_export):
    info = extinction_export["info"]
    assert info["task"] == "regression"
    assert info["prediction_output"] == "extinction_diameter_ratio"
    assert info["prediction_bounds"] == [0.0, 1.0]
    assert info["scaled_outputs"] == [{"name": "extinction_diameter_mm", "input": "droplet_d0_mm"}]
    assert info["outputs"][:2] == ["extinction_diameter_ratio", "extinction_diameter_mm"]
    assert "high_confidence_min_probability" not in info["support"]["confidence"]
    assert "class_labels" not in info
    assert sorted(info["training_source"]["target"]) == ["max", "mean", "min", "std"]
