from __future__ import annotations

import json
import subprocess
import sys

import pytest

from flame.training.evaluate import evaluate_models
from flame.training.export import export_from_data, round_outward
from flame.training.reporting import display_path, evaluation_report

INFO_FIELDS = ["model_version", "task", "algorithm", "class_labels", "positive_class", "decision_threshold", "inputs",
               "input_mapping", "constraints", "fixed_inputs", "feature_columns", "feature_dtypes", "outputs",
               "training_source", "support", "cv", "decisions", "config_sha256", "versions", "code_version", "files",
               "created"]

LOAD_WITHOUT_TRAINING_CODE = """
import json, sys
import joblib
import pandas as pd
info = json.load(open("model_info.json", encoding="utf-8"))
model = joblib.load("model.joblib")
support = joblib.load("support.joblib")
row = pd.DataFrame([{"fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "co2": 0.0, "he": 0.0, "d0_mm": 3.0,
                     "fiber": False}])[info["feature_columns"]]
probabilities = model.predict_proba(row)
partition = support["partitions"]["methanol"]
distances, _ = partition["neighbors"].kneighbors(partition["scaler"].transform(row[support["columns"]].to_numpy()))
assert probabilities.shape == (1, 2) and distances.shape[0] == 1
assert not any(name == "flame" or name.startswith("flame.") for name in sys.modules)
print("loaded")
"""


@pytest.mark.parametrize(("value", "upward", "expected"), [
    (0.34, True, 0.34),
    (0.34, False, 0.34),
    (0.6925, False, 0.69),
    (1.01855, True, 1.02),
    (1.1, False, 1.1),
    (4.67, True, 4.67),
])
def test_ranges_are_rounded_outward_without_float_artifacts(value, upward, expected):
    assert round_outward(value, 2, upward) == expected


def test_artifact_directory_has_every_file(synthetic_export):
    names = sorted(path.name for path in synthetic_export["artifact_dir"].iterdir())
    assert names == ["model.joblib", "model_info.json", "support.joblib", "support_reference.csv"]
    assert synthetic_export["artifact_dir"].name == "flex-outcome-0.2"


def test_model_info_has_the_required_fields(synthetic_export):
    info = json.loads((synthetic_export["artifact_dir"] / "model_info.json").read_text(encoding="utf-8"))
    assert list(info) == INFO_FIELDS
    assert info["algorithm"]["name"] == synthetic_export["selected"]
    assert info["cv"]["baseline"] == "logreg"
    assert sorted(info["versions"]) == ["joblib", "numpy", "pandas", "python", "scikit-learn", "scipy"]
    assert info["support"]["nearest_test_conditions"] == ["fuel", "pressure_atm", "o2", "co2", "he", "d0_mm"]
    assert sorted(info["support"]["thresholds"]) == ["heptane", "methanol"]
    assert info["constraints"] == [{"kind": "max_sum", "columns": ["o2", "co2", "he"], "max": 1.0}]


def test_model_info_traces_the_data_the_config_and_the_decisions(synthetic_export, synthetic_data, outcome_config):
    info = synthetic_export["info"]
    source = info["training_source"]
    assert source["sha256"] == synthetic_data.source_sha256
    assert source["rows"] == len(synthetic_data.rows)
    assert source["exclusions"] == synthetic_data.exclusions
    assert source["target"] == synthetic_data.target.value_counts().to_dict()
    assert info["config_sha256"] == outcome_config.sha256
    assert info["decisions"] == outcome_config.decisions


def test_input_ranges_cover_the_training_rows(synthetic_export, synthetic_data):
    inputs = synthetic_export["info"]["inputs"]
    features = synthetic_data.features
    for name, column in (("pressure_atm", "pressure_atm"), ("o2", "o2"), ("droplet_d0_mm", "d0_mm")):
        assert inputs[name]["min"] <= features[column].min() <= features[column].max() <= inputs[name]["max"]
        assert inputs[name]["min"] > features[column].min() - 0.01
        assert inputs[name]["max"] < features[column].max() + 0.01
    assert inputs["fuel"]["values"] == ["heptane", "methanol"]
    assert inputs["suppressant_fraction"]["by_selector"]["none"] == {"min": 0.0, "max": 0.0}


def test_support_reference_lists_every_training_row_in_order(synthetic_export, synthetic_data):
    reference = synthetic_export["bundle"]["reference"]
    assert list(reference["cir_id"]) == list(synthetic_data.rows["cir_id"])
    assert list(reference.columns[:3]) == ["cir_id", "testsheet_id", "test_end"]


def test_artifact_loads_in_a_clean_interpreter_without_the_training_code(synthetic_export):
    completed = subprocess.run([sys.executable, "-I", "-c", LOAD_WITHOUT_TRAINING_CODE],
                               cwd=synthetic_export["artifact_dir"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "loaded"


def test_reports_are_written_next_to_the_artifact(synthetic_export):
    card = synthetic_export["model_card"].read_text(encoding="utf-8")
    evaluation = synthetic_export["evaluation_report"].read_text(encoding="utf-8")
    assert synthetic_export["model_card"].name == "flex-outcome-0.2_model_card.md"
    assert synthetic_export["evaluation_report"].name == "flex-outcome_evaluation.md"
    assert "## Limitations" in card and "## Reference queries" in card and "## Reproducibility" in card
    assert "## Paired comparison with the baseline" in evaluation


def test_reports_and_model_info_are_utf8_with_unix_line_endings(synthetic_export):
    paths = [synthetic_export["model_card"], synthetic_export["evaluation_report"],
             synthetic_export["artifact_dir"] / "model_info.json"]
    texts = [path.read_bytes().decode("utf-8") for path in paths]
    assert not any("\r" in text for text in texts)
    assert "mean ± std" in texts[1]


def test_report_paths_use_forward_slashes(outcome_config):
    assert display_path(outcome_config.path) == "configs/training/flex_outcome.yml"


def test_sensitivity_config_is_never_exported(synthetic_table, synthetic_data, sensitivity_config, tmp_path):
    with pytest.raises(ValueError, match="only publishable"):
        export_from_data(sensitivity_config, synthetic_data, synthetic_table, tmp_path / "a", tmp_path / "r")
    assert not (tmp_path / "a").exists()


def test_two_evaluations_give_identical_metrics_and_reports(synthetic_data, outcome_config):
    first = evaluate_models(synthetic_data, outcome_config)
    second = evaluate_models(synthetic_data, outcome_config)
    assert first.fold_metrics.equals(second.fold_metrics)
    assert first.repeat_metrics.equals(second.repeat_metrics)
    assert first.selection == second.selection
    assert (evaluation_report(outcome_config, synthetic_data, first)
            == evaluation_report(outcome_config, synthetic_data, second))


def test_every_model_is_scored_on_every_fold(synthetic_data, outcome_config):
    evaluation = evaluate_models(synthetic_data, outcome_config)
    cv = outcome_config.cv
    counts = evaluation.fold_metrics.groupby("model").size()
    assert counts.to_dict() == {name: cv.n_splits * cv.n_repeats for name in outcome_config.models}
    assert evaluation.metric_names == ["macro_f1", "recall_extinction", "recall_no_extinction", "brier", "roc_auc"]
    per_repeat = evaluation.predictions.groupby(["model", "repeat"])["row"].nunique()
    assert (per_repeat == len(synthetic_data.rows)).all()
