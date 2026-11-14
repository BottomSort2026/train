from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flame.training.support import confidence_level, query_support

MODEL_FILE = "model.joblib"
SUPPORT_FILE = "support.joblib"
REFERENCE_FILE = "support_reference.csv"
INFO_FILE = "model_info.json"
FEATURE_DTYPES = {"str": str, "float": float, "bool": bool}
SUM_TOLERANCE = 1e-9


def load_bundle(artifact_dir: str | Path) -> dict[str, Any]:
    directory = Path(artifact_dir)
    return {
        "model": joblib.load(directory / MODEL_FILE),
        "support": joblib.load(directory / SUPPORT_FILE),
        "reference": pd.read_csv(directory / REFERENCE_FILE, dtype={"testsheet_id": "string"}),
        "info": json.loads((directory / INFO_FILE).read_text()),
    }


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_inputs(info: dict[str, Any], inputs: dict[str, Any]) -> list[str]:
    spec = info["inputs"]
    missing = [name for name in spec if name not in inputs]
    unexpected = [name for name in inputs if name not in spec]
    if missing or unexpected:
        raise ValueError(f"Inputs must be exactly {list(spec)}; missing {missing}, unexpected {unexpected}")
    selector = info["input_mapping"]["selector"]
    for name, field in spec.items():
        if field["type"] == "enum" and inputs[name] not in field["values"]:
            raise ValueError(f"{name} must be one of {field['values']}, got {inputs[name]!r}")
        if field["type"] != "enum" and not is_number(inputs[name]):
            raise ValueError(f"{name} must be a finite number, got {inputs[name]!r}")
    out_of_range = []
    for name, field in spec.items():
        if field["type"] == "enum":
            continue
        limits = field["by_selector"][inputs[selector["input"]]] if name == selector["value_input"] else field
        if not limits["min"] <= inputs[name] <= limits["max"]:
            out_of_range.append(name)
    if inputs[selector["input"]] == selector["empty_value"] and inputs[selector["value_input"]] != 0:
        raise ValueError(f"{selector['value_input']} must be 0 when {selector['input']} is "
                         f"{selector['empty_value']!r}")
    return out_of_range


def build_feature_row(info: dict[str, Any], inputs: dict[str, Any]) -> pd.DataFrame:
    mapping = info["input_mapping"]
    selector = mapping["selector"]
    row = {column: inputs[name] for name, column in mapping["direct"].items()}
    for selected_value, column in selector["columns"].items():
        chosen = inputs[selector["input"]] == selected_value
        row[column] = float(inputs[selector["value_input"]]) if chosen else 0.0
    row.update(info["fixed_inputs"])
    frame = pd.DataFrame([row])[info["feature_columns"]]
    return frame.astype({name: FEATURE_DTYPES[kind] for name, kind in info["feature_dtypes"].items()})


def check_constraints(info: dict[str, Any], feature_row: pd.DataFrame) -> None:
    for constraint in info["constraints"]:
        total = float(feature_row[constraint["columns"]].iloc[0].sum())
        if total > constraint["max"] + SUM_TOLERANCE:
            raise ValueError(f"{' + '.join(constraint['columns'])} must be <= {constraint['max']}, got {total:g}")


def predict_outputs(bundle: dict[str, Any], feature_row: pd.DataFrame,
                    inputs: dict[str, Any]) -> tuple[dict[str, Any], float | None]:
    info, model = bundle["info"], bundle["model"]
    if info["task"] == "classification":
        probabilities = {str(label): float(value)
                         for label, value in zip(model.classes_, model.predict_proba(feature_row)[0])}
        positive = info["positive_class"]
        negative = next(label for label in info["class_labels"] if label != positive)
        outcome = positive if probabilities[positive] >= info["decision_threshold"] else negative
        return {"outcome": outcome, "probabilities": probabilities}, probabilities[outcome]
    value = float(model.predict(feature_row)[0])
    bounds = info.get("prediction_bounds")
    if bounds:
        value = float(np.clip(value, bounds[0], bounds[1]))
    outputs = {info["prediction_output"]: value}
    for scaled in info.get("scaled_outputs", []):
        outputs[scaled["name"]] = value * float(inputs[scaled["input"]])
    return outputs, None


def nearest_tests(bundle: dict[str, Any], support: dict[str, Any]) -> list[dict[str, Any]]:
    reference = bundle["reference"]
    fields = bundle["info"]["support"]["nearest_test_fields"]
    tests = []
    for row, distance in zip(support["nearest_reference_rows"], support["nearest_distances"]):
        record = reference.iloc[row]
        test = {field: (None if pd.isna(record[field]) else str(record[field])) for field in fields}
        tests.append({**test, "distance": distance})
    return tests


def reference_predict(bundle: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    info = bundle["info"]
    out_of_range_fields = validate_inputs(info, inputs)
    feature_row = build_feature_row(info, inputs)
    check_constraints(info, feature_row)
    outputs, top_probability = predict_outputs(bundle, feature_row, inputs)
    support = query_support(bundle["support"], feature_row)
    in_tested_range = support["within_joint_range"] and not out_of_range_fields
    source = info["training_source"]
    return {
        **outputs,
        "in_tested_range": in_tested_range,
        "out_of_range_fields": out_of_range_fields,
        "support_n": support["support_n"],
        "nearest_tests": nearest_tests(bundle, support),
        "confidence": confidence_level(in_tested_range, support["support_n"], top_probability,
                                       info["support"]["confidence"]),
        "model_version": info["model_version"],
        "training_source": {key: source[key] for key in ("dataset", "accession", "sha256", "rows")},
    }
