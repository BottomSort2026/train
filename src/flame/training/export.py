from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import date
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn

from configs.training.config_training import ExperimentConfig
from flame.training.datasets import TrainingData, build_training_data, load_source_table
from flame.training.evaluate import SELECTION_UNIT, Evaluation, evaluate_models
from flame.training.flex_data import TRAIN_ROOT
from flame.training.inference import INFO_FILE, MODEL_FILE, REFERENCE_FILE, SUPPORT_FILE, load_bundle
from flame.training.inference import reference_predict
from flame.training.models import ESTIMATORS, fit_model, is_calibrated
from flame.training.reporting import model_card, write_evaluation_report
from flame.training.support import fit_support, support_thresholds

ARTIFACTS_ROOT = TRAIN_ROOT / "artifacts" / "models"
REPORTS_ROOT = TRAIN_ROOT / "reports"
COMMON_OUTPUTS = ["in_tested_range", "out_of_range_fields", "support_n", "nearest_tests", "confidence",
                  "model_version", "training_source"]


def round_outward(value: float, decimals: int, upward: bool) -> float:
    quantum = Decimal(1).scaleb(-decimals)
    rounding = ROUND_CEILING if upward else ROUND_FLOOR
    return float(Decimal(repr(float(value))).quantize(quantum, rounding=rounding))


def tested_range(values: pd.Series, decimals: int) -> dict[str, float]:
    return {"min": round_outward(values.min(), decimals, upward=False),
            "max": round_outward(values.max(), decimals, upward=True)}


def describe_inputs(config: ExperimentConfig, data: TrainingData) -> dict[str, Any]:
    interface = config.interface
    decimals = interface.range_decimals
    fields: dict[str, Any] = {}
    for item in interface.direct_inputs:
        column = data.features[item.column]
        if item.type == "enum":
            fields[item.name] = {"type": "enum", "values": sorted(str(value) for value in column.unique())}
        else:
            fields[item.name] = {"type": "float", **tested_range(column, decimals), "unit": item.unit}
    selector = interface.selector
    selected = data.rows[selector.source_column].astype(str)
    known = {selector.empty_value, *selector.columns}
    if not set(selected.unique()) <= known:
        raise ValueError(f"{selector.source_column} has values outside {sorted(known)}")
    by_selector: dict[str, dict[str, float]] = {}
    if (selected == selector.empty_value).any():
        by_selector[selector.empty_value] = {"min": 0.0, "max": 0.0}
    for value, column in selector.columns.items():
        if (selected == value).any():
            by_selector[value] = tested_range(data.features.loc[selected == value, column], decimals)
    fields[selector.input] = {"type": "enum", "values": list(by_selector)}
    fields[selector.value_input] = {
        "type": "float",
        "min": min(limits["min"] for limits in by_selector.values()),
        "max": max(limits["max"] for limits in by_selector.values()),
        "unit": selector.unit,
        "by_selector": by_selector,
    }
    return {name: fields[name] for name in interface.input_order}


def describe_input_mapping(config: ExperimentConfig) -> dict[str, Any]:
    interface = config.interface
    selector = interface.selector
    return {
        "direct": {item.name: item.column for item in interface.direct_inputs},
        "selector": {"input": selector.input, "value_input": selector.value_input,
                     "empty_value": selector.empty_value, "columns": dict(selector.columns)},
    }


def feature_dtypes(config: ExperimentConfig) -> dict[str, str]:
    features = config.features
    groups = (("str", features.categorical), ("float", features.numeric), ("bool", features.boolean))
    return {column: kind for kind, columns in groups for column in columns}


def describe_task(config: ExperimentConfig) -> dict[str, Any]:
    target = config.target
    if config.task == "classification":
        return {
            "class_labels": target.class_labels,
            "positive_class": target.positive_class,
            "decision_threshold": target.decision_threshold,
            "outputs": ["outcome", "probabilities", *COMMON_OUTPUTS],
        }
    scaled = [{"name": item.name, "input": item.input} for item in target.scaled_outputs]
    return {
        "prediction_output": target.output,
        "prediction_bounds": list(target.prediction_bounds) if target.prediction_bounds else None,
        "scaled_outputs": scaled,
        "outputs": [target.output, *[item["name"] for item in scaled], *COMMON_OUTPUTS],
    }


def describe_algorithm(config: ExperimentConfig, model_name: str) -> dict[str, Any]:
    model = config.models[model_name]
    spline = model.spline
    calibration = config.calibration
    return {
        "name": model_name,
        "estimator": ESTIMATORS[model.estimator].__name__,
        "params": dict(model.params),
        "spline": ({"columns": list(spline.columns), "n_knots": spline.n_knots, "degree": spline.degree}
                   if spline else None),
        "calibrated": is_calibrated(model),
        "calibration": ({"method": calibration.method, "n_splits": calibration.n_splits}
                        if is_calibrated(model) else None),
    }


def describe_target(config: ExperimentConfig, data: TrainingData) -> dict[str, Any]:
    if config.task == "classification":
        return {name: int(count) for name, count in data.target.value_counts().items()}
    stats = data.target.describe()
    return {name: float(stats[name]) for name in ("min", "max", "mean", "std")}


def describe_training_source(config: ExperimentConfig, data: TrainingData) -> dict[str, Any]:
    dataset = config.dataset
    target_column = config.target.column
    source = {
        "dataset": dataset.name,
        "accession": dataset.accession,
        "sha256": data.source_sha256,
        "source_rows": data.source_rows,
        "rows": int(len(data.rows)),
        "groups": int(data.groups.nunique()),
        "target": describe_target(config, data),
        "exclusions": dict(data.exclusions),
    }
    if config.task == "classification":
        source["original_target_values"] = {str(name): int(count)
                                            for name, count in data.rows[target_column].value_counts().items()}
    return source


def describe_cv(config: ExperimentConfig, evaluation: Evaluation) -> dict[str, Any]:
    cv = config.cv
    metrics = {
        model: {metric: {"mean": float(rows[metric].mean()), "std": float(rows[metric].std(ddof=0))}
                for metric in evaluation.metric_names}
        for model, rows in evaluation.repeat_metrics.groupby("model", sort=False)
    }
    return {
        "splitter": "StratifiedGroupKFold",
        "group_column": config.dataset.group_column,
        "n_splits": cv.n_splits,
        "n_repeats": cv.n_repeats,
        "shuffle": cv.shuffle,
        "seed": cv.seed,
        "unit": SELECTION_UNIT,
        "baseline": evaluation.selection["baseline"],
        "selected": evaluation.selection["selected"],
        "metrics": metrics,
        "selection": evaluation.selection,
    }


def describe_support(config: ExperimentConfig, data: TrainingData, support_bundle: dict[str, Any]) -> dict[str, Any]:
    support = config.support
    return {
        "columns": list(support.columns),
        "partition_column": support.partition_column,
        "range_percentile": support.range_percentile,
        "nearest_k": support.nearest_k,
        "thresholds": support_thresholds(support_bundle),
        "nearest_test_fields": list(data.id_columns),
        "nearest_test_conditions": [support.partition_column, *support.columns],
        "confidence": support.confidence_rule(),
    }


def code_version() -> dict[str, Any] | None:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=TRAIN_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=TRAIN_ROOT, capture_output=True, text=True,
                                check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return {"commit": commit, "dirty": bool(status)}


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def build_model_info(config: ExperimentConfig, data: TrainingData, evaluation: Evaluation,
                     support_bundle: dict[str, Any]) -> dict[str, Any]:
    interface = config.interface
    composition = interface.composition
    task = describe_task(config)
    outputs = task.pop("outputs")
    return json_ready({
        "model_version": config.model_version,
        "task": config.task,
        "algorithm": describe_algorithm(config, evaluation.selection["selected"]),
        **task,
        "inputs": describe_inputs(config, data),
        "input_mapping": describe_input_mapping(config),
        "constraints": [{"kind": "max_sum", "columns": list(composition.columns), "max": composition.max_sum}],
        "fixed_inputs": dict(interface.fixed_inputs),
        "feature_columns": config.feature_columns,
        "feature_dtypes": feature_dtypes(config),
        "outputs": outputs,
        "training_source": describe_training_source(config, data),
        "support": describe_support(config, data, support_bundle),
        "cv": describe_cv(config, evaluation),
        "decisions": config.decisions,
        "config_sha256": config.sha256,
        "versions": library_versions(),
        "code_version": code_version(),
        "files": {"model": MODEL_FILE, "support": SUPPORT_FILE, "support_reference": REFERENCE_FILE},
        "created": date.today().isoformat(),
    })


def support_reference(config: ExperimentConfig, data: TrainingData) -> pd.DataFrame:
    support = config.support
    return pd.concat([data.ids, data.features[[support.partition_column, *support.columns]]], axis=1)


def write_artifact(artifact_dir: Path, pipeline: Any, support_bundle: dict[str, Any], reference: pd.DataFrame,
                   info: dict[str, Any]) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_dir / MODEL_FILE)
    joblib.dump(support_bundle, artifact_dir / SUPPORT_FILE)
    reference.to_csv(artifact_dir / REFERENCE_FILE, index=False, lineterminator="\n")
    (artifact_dir / INFO_FILE).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8", newline="\n")


def run_reference_queries(bundle: dict[str, Any], config: ExperimentConfig) -> list[dict[str, Any]]:
    return [{"name": query.name, "inputs": dict(query.inputs), "result": reference_predict(bundle, dict(query.inputs))}
            for query in config.report.reference_queries]


def export_from_data(config: ExperimentConfig, data: TrainingData, source_table: pd.DataFrame,
                     artifacts_root: Path = ARTIFACTS_ROOT, reports_root: Path = REPORTS_ROOT) -> dict[str, Any]:
    if config.role != "publishable":
        raise ValueError(f"{config.name} has role {config.role!r}: only publishable experiments are exported")
    evaluation = evaluate_models(data, config)
    evaluation_path = write_evaluation_report(config, data, evaluation, reports_root)
    selected = evaluation.selection["selected"]
    pipeline = fit_model(selected, config, data.features, data.target, data.groups, config.cv.seed)
    support_bundle = fit_support(data.features, config.support)
    info = build_model_info(config, data, evaluation, support_bundle)
    artifact_dir = artifacts_root / config.model_version
    write_artifact(artifact_dir, pipeline, support_bundle, support_reference(config, data), info)
    bundle = load_bundle(artifact_dir)
    reference_results = run_reference_queries(bundle, config)
    card_path = reports_root / f"{config.model_version}_model_card.md"
    card_path.write_text(model_card(config, data, evaluation, info, reference_results, source_table),
                         encoding="utf-8", newline="\n")
    return {"artifact_dir": artifact_dir, "model_card": card_path, "evaluation_report": evaluation_path,
            "selected": selected, "info": info, "reference_results": reference_results}


def export_experiment(config: ExperimentConfig, artifacts_root: Path = ARTIFACTS_ROOT,
                      reports_root: Path = REPORTS_ROOT) -> dict[str, Any]:
    source_table, source_sha256 = load_source_table(config)
    data = build_training_data(source_table, config, source_sha256)
    return export_from_data(config, data, source_table, artifacts_root, reports_root)
