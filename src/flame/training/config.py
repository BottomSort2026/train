from __future__ import annotations

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flame.training.flex_data import TRAIN_ROOT

TASKS = ("classification", "regression")
ROLES = ("publishable", "sensitivity")
MODEL_ROLES = ("baseline", "candidate")
REQUIRED_SECTIONS = ("experiment", "dataset", "target", "features", "models", "cv", "selection", "support",
                     "interface", "report")


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    sha256: str
    values: dict[str, Any]
    reference: ExperimentConfig | None = None

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]

    @property
    def name(self) -> str:
        return self.values["experiment"]["name"]

    @property
    def version(self) -> str:
        return self.values["experiment"]["version"]

    @property
    def model_version(self) -> str:
        return f"{self.name}-{self.version}"

    @property
    def role(self) -> str:
        return self.values["experiment"]["role"]

    @property
    def task(self) -> str:
        return self.values["experiment"]["task"]

    @property
    def decisions(self) -> list[str]:
        return list(self.values["experiment"]["decisions"])

    @property
    def baseline_name(self) -> str:
        return next(name for name, model in self.values["models"].items() if model["role"] == "baseline")

    @property
    def candidate_names(self) -> list[str]:
        return [name for name, model in self.values["models"].items() if model["role"] == "candidate"]

    @property
    def feature_columns(self) -> list[str]:
        features = self.values["features"]
        return [*features["categorical"], *features["numeric"], *features["boolean"]]


def resolve_path(relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else TRAIN_ROOT / path


def _merge_sections(reference: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = {name: dict(section) for name, section in reference.items()}
    for name, section in overrides.items():
        merged[name] = {**merged.get(name, {}), **section}
    return merged


def validate_config(values: dict[str, Any]) -> None:
    missing = [name for name in REQUIRED_SECTIONS if name not in values]
    if missing:
        raise ValueError(f"Config is missing sections: {missing}")
    experiment = values["experiment"]
    if experiment["role"] not in ROLES:
        raise ValueError(f"experiment.role must be one of {ROLES}, got {experiment['role']!r}")
    if experiment["task"] not in TASKS:
        raise ValueError(f"experiment.task must be one of {TASKS}, got {experiment['task']!r}")
    roles = [model["role"] for model in values["models"].values()]
    if any(role not in MODEL_ROLES for role in roles):
        raise ValueError(f"Every model role must be one of {MODEL_ROLES}, got {roles}")
    if roles.count("baseline") != 1:
        raise ValueError(f"Exactly one model must have role 'baseline', found {roles.count('baseline')}")
    calibrated = [name for name, model in values["models"].items() if model.get("calibrated", False)]
    if calibrated and "calibration" not in values:
        raise ValueError(f"Models {calibrated} are calibrated but the config has no [calibration] section")
    if experiment["task"] == "classification":
        classes = set(values["target"].get("class_map", {}).values())
        if len(classes) != 2 or values["target"].get("positive_class") not in classes:
            raise ValueError("A classification config needs [target.class_map] with two classes, one of them "
                             "being target.positive_class")
    if experiment["task"] == "regression":
        if "output" not in values["target"] or "stratify_bins" not in values["cv"]:
            raise ValueError("A regression config needs target.output and cv.stratify_bins")
    if experiment["role"] == "sensitivity" and "comparison" not in values:
        raise ValueError("A sensitivity config needs a [comparison] section")


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = resolve_path(path)
    content = config_path.read_bytes().replace(b"\r\n", b"\n")
    values = tomllib.loads(content.decode("utf-8"))
    reference = None
    reference_path = values.get("comparison", {}).get("reference_config")
    if reference_path is not None:
        reference = load_config(reference_path)
        values = _merge_sections(reference.values, values)
    validate_config(values)
    return ExperimentConfig(path=config_path, sha256=hashlib.sha256(content).hexdigest(), values=values,
                            reference=reference)
