"""Loads every experiment YAML in configs/training/ into frozen dataclasses, once, at import time.

Usage:
    from configs.training.config_training import get_experiment, training_settings
    config = get_experiment("flex-outcome")      # by experiment name, or by the path of its YAML file
    config.cv.n_splits
    config.models["hgb"].params

One YAML file per experiment. A sensitivity experiment names its reference experiment in
`comparison.reference_config`: it inherits every section of that file, and its own sections replace keys one
level deep (decision T-02). Unknown keys and invalid values fail at load time.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from configs import PROJECT_ROOT

TRAINING_CONFIG_DIR = PROJECT_ROOT / "configs" / "training"

TASKS = ("classification", "regression")
ROLES = ("publishable", "sensitivity")
MODEL_ROLES = ("baseline", "candidate")
EXCLUSION_RULES = ("not_null", "is_null", "is_true", "equals", "not_equals")
RULES_WITH_VALUE = ("equals", "not_equals")
CLASSIFIERS = ("logistic_regression", "hist_gradient_boosting_classifier")
REGRESSORS = ("ridge", "hist_gradient_boosting_regressor")
ESTIMATOR_NAMES = CLASSIFIERS + REGRESSORS
DIRECTIONS = ("higher", "lower")
CALIBRATION_METHODS = ("sigmoid", "isotonic")
INPUT_TYPES = ("enum", "float")


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _check_keys(section: Any, where: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> dict:
    """Fail on unknown keys (typos) and on missing required keys."""
    if not isinstance(section, dict):
        raise ValueError(f"{where} must be a mapping, got {type(section).__name__}")
    unknown = set(section) - set(required) - set(optional)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")
    missing = [key for key in required if key not in section]
    if missing:
        raise ValueError(f"Missing keys in {where}: {missing}")
    return section


def _strings(values: Any, where: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{where} must be a list of strings, got {values!r}")
    return tuple(values)


def _string_map(values: Any, where: str) -> dict[str, str]:
    if not isinstance(values, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in values.items()):
        raise ValueError(f"{where} must map strings to strings, got {values!r}")
    return dict(values)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_number(value: Any, where: str, low: float | None = None, high: float | None = None) -> None:
    if not _is_number(value):
        raise ValueError(f"{where} must be a number, got {value!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{where} must be between {low} and {high}, got {value!r}")


def _check_int(value: Any, where: str, low: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < low:
        raise ValueError(f"{where} must be an integer >= {low}, got {value!r}")


def _check_choice(value: Any, where: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(f"{where} must be one of {choices}, got {value!r}")


def _check_scalars(values: dict[str, Any], where: str) -> None:
    for key, value in values.items():
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"{where}.{key} must be a string, number or boolean, got {value!r}")


def resolve_path(relative: str | Path) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Dataclasses, one per YAML section
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Experiment:
    name: str
    version: str
    role: str
    task: str
    decisions: tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.version, str):
            raise ValueError(f"experiment.version must be a quoted string, got {self.version!r}")
        _check_choice(self.role, "experiment.role", ROLES)
        _check_choice(self.task, "experiment.task", TASKS)


@dataclass(frozen=True)
class Exclusion:
    name: str
    column: str
    rule: str
    value: Any = None

    def __post_init__(self):
        _check_choice(self.rule, f"exclusion {self.name!r} rule", EXCLUSION_RULES)
        if self.rule in RULES_WITH_VALUE and self.value is None:
            raise ValueError(f"Exclusion {self.name!r} with rule {self.rule!r} needs a value")


@dataclass(frozen=True)
class Dataset:
    name: str
    accession: str
    path: Path
    group_column: str
    row_key: str
    id_columns: tuple[str, ...]
    exclusions: tuple[Exclusion, ...]
    extra_exclusions: tuple[Exclusion, ...] = ()

    def __post_init__(self):
        names = [item.name for item in (*self.exclusions, *self.extra_exclusions)]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Repeated exclusion names in dataset: {repeated}")

    @property
    def all_exclusions(self) -> tuple[Exclusion, ...]:
        return (*self.exclusions, *self.extra_exclusions)


@dataclass(frozen=True)
class ScaledOutput:
    name: str
    input: str


@dataclass(frozen=True)
class Target:
    column: str
    positive_class: str | None = None
    decision_threshold: float | None = None
    class_map: dict[str, str] | None = None
    output: str | None = None
    prediction_bounds: tuple[float, float] | None = None
    scaled_outputs: tuple[ScaledOutput, ...] = ()

    def __post_init__(self):
        if self.decision_threshold is not None:
            _check_number(self.decision_threshold, "target.decision_threshold", 0.0, 1.0)
        if self.prediction_bounds is not None:
            if len(self.prediction_bounds) != 2 or not all(_is_number(v) for v in self.prediction_bounds) \
                    or self.prediction_bounds[0] >= self.prediction_bounds[1]:
                raise ValueError(f"target.prediction_bounds must be [low, high], got {self.prediction_bounds!r}")

    @property
    def class_labels(self) -> list[str]:
        return sorted(set((self.class_map or {}).values()))


@dataclass(frozen=True)
class Product:
    name: str
    columns: tuple[str, ...]


@dataclass(frozen=True)
class Features:
    categorical: tuple[str, ...]
    numeric: tuple[str, ...]
    boolean: tuple[str, ...]
    products: tuple[Product, ...] = ()

    def __post_init__(self):
        columns = self.columns
        repeated = sorted({column for column in columns if columns.count(column) > 1})
        if repeated:
            raise ValueError(f"Feature columns listed twice: {repeated}")
        for product in self.products:
            unknown = set(product.columns) - set(self.numeric)
            if unknown:
                raise ValueError(f"features.products {product.name!r} uses non-numeric columns {sorted(unknown)}")

    @property
    def columns(self) -> list[str]:
        return [*self.categorical, *self.numeric, *self.boolean]


@dataclass(frozen=True)
class Spline:
    columns: tuple[str, ...]
    n_knots: int
    degree: int

    def __post_init__(self):
        _check_int(self.n_knots, "spline.n_knots", 2)
        _check_int(self.degree, "spline.degree", 1)


@dataclass(frozen=True)
class Model:
    name: str
    role: str
    estimator: str
    params: dict[str, Any] = field(default_factory=dict)
    calibrated: bool = False
    spline: Spline | None = None

    def __post_init__(self):
        _check_choice(self.role, f"models.{self.name}.role", MODEL_ROLES)
        _check_choice(self.estimator, f"models.{self.name}.estimator", ESTIMATOR_NAMES)
        if not isinstance(self.calibrated, bool):
            raise ValueError(f"models.{self.name}.calibrated must be true or false")
        _check_scalars(self.params, f"models.{self.name}.params")
        if self.calibrated and self.estimator not in CLASSIFIERS:
            raise ValueError(f"models.{self.name}: only classifiers can be calibrated")


@dataclass(frozen=True)
class CrossValidation:
    n_splits: int
    n_repeats: int
    shuffle: bool
    seed: int
    stratify_bins: int | None = None

    def __post_init__(self):
        _check_int(self.n_splits, "cv.n_splits", 2)
        _check_int(self.n_repeats, "cv.n_repeats", 1)
        _check_int(self.seed, "cv.seed", 0)
        if not isinstance(self.shuffle, bool):
            raise ValueError("cv.shuffle must be true or false")
        if self.stratify_bins is not None:
            _check_int(self.stratify_bins, "cv.stratify_bins", 2)


@dataclass(frozen=True)
class Calibration:
    method: str
    n_splits: int

    def __post_init__(self):
        _check_choice(self.method, "calibration.method", CALIBRATION_METHODS)
        _check_int(self.n_splits, "calibration.n_splits", 2)


@dataclass(frozen=True)
class MetricRule:
    metric: str
    direction: str

    def __post_init__(self):
        _check_choice(self.direction, f"direction of {self.metric!r}", DIRECTIONS)


@dataclass(frozen=True)
class Selection:
    tie_breaker_metric: str
    tie_breaker_direction: str
    must_beat: tuple[MetricRule, ...]

    def __post_init__(self):
        _check_choice(self.tie_breaker_direction, "selection.tie_breaker_direction", DIRECTIONS)
        if not self.must_beat:
            raise ValueError("selection.must_beat needs at least one metric")


@dataclass(frozen=True)
class Support:
    columns: tuple[str, ...]
    partition_column: str
    range_percentile: float
    nearest_k: int
    low_confidence_below_support: int
    high_confidence_min_support: int
    high_confidence_min_probability: float | None = None

    def __post_init__(self):
        _check_number(self.range_percentile, "support.range_percentile", 0, 100)
        _check_int(self.nearest_k, "support.nearest_k", 1)
        _check_int(self.low_confidence_below_support, "support.low_confidence_below_support", 0)
        _check_int(self.high_confidence_min_support, "support.high_confidence_min_support", 0)
        if self.high_confidence_min_support < self.low_confidence_below_support:
            raise ValueError("support.high_confidence_min_support must be >= support.low_confidence_below_support")
        if self.high_confidence_min_probability is not None:
            _check_number(self.high_confidence_min_probability, "support.high_confidence_min_probability", 0.0, 1.0)

    def confidence_rule(self) -> dict[str, Any]:
        """The confidence thresholds, as written to model_info.json (the probability one only when set)."""
        rule = {"low_confidence_below_support": self.low_confidence_below_support,
                "high_confidence_min_support": self.high_confidence_min_support}
        if self.high_confidence_min_probability is not None:
            rule["high_confidence_min_probability"] = self.high_confidence_min_probability
        return rule


@dataclass(frozen=True)
class Composition:
    columns: tuple[str, ...]
    max_sum: float

    def __post_init__(self):
        _check_number(self.max_sum, "interface.composition.max_sum", 0.0)


@dataclass(frozen=True)
class Selector:
    input: str
    value_input: str
    source_column: str
    empty_value: str
    unit: str
    columns: dict[str, str]

    def __post_init__(self):
        if self.empty_value in self.columns:
            raise ValueError("interface.selector.columns must not contain the empty value")


@dataclass(frozen=True)
class DirectInput:
    name: str
    column: str
    type: str
    unit: str | None = None

    def __post_init__(self):
        _check_choice(self.type, f"interface input {self.name!r} type", INPUT_TYPES)
        if self.type == "float" and self.unit is None:
            raise ValueError(f"interface input {self.name!r} is a float and needs a unit")


@dataclass(frozen=True)
class Interface:
    range_decimals: int
    input_order: tuple[str, ...]
    fixed_inputs: dict[str, Any]
    composition: Composition
    selector: Selector
    direct_inputs: tuple[DirectInput, ...]

    def __post_init__(self):
        _check_int(self.range_decimals, "interface.range_decimals", 0)
        _check_scalars(self.fixed_inputs, "interface.fixed_inputs")
        exposed = [item.name for item in self.direct_inputs] + [self.selector.input, self.selector.value_input]
        if sorted(exposed) != sorted(self.input_order) or len(set(self.input_order)) != len(self.input_order):
            raise ValueError(f"interface.input_order must list each exposed input once: {sorted(exposed)}")


@dataclass(frozen=True)
class EvaluationSettings:
    region_columns: tuple[str, ...] = ()
    region_round: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReferenceQuery:
    name: str
    inputs: dict[str, Any]


@dataclass(frozen=True)
class Report:
    pressure_level_decimals: int
    intended_use: str
    limitations: tuple[str, ...]
    reference_queries: tuple[ReferenceQuery, ...] = ()


@dataclass(frozen=True)
class Comparison:
    reference_config: Path
    max_abs_roc_auc_delta: float
    min_spearman: float

    def __post_init__(self):
        _check_number(self.max_abs_roc_auc_delta, "comparison.max_abs_roc_auc_delta", 0.0, 1.0)
        _check_number(self.min_spearman, "comparison.min_spearman", -1.0, 1.0)


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    sha256: str
    experiment: Experiment
    dataset: Dataset
    target: Target
    features: Features
    models: dict[str, Model]
    cv: CrossValidation
    selection: Selection
    support: Support
    interface: Interface
    report: Report
    calibration: Calibration | None = None
    evaluation: EvaluationSettings | None = None
    comparison: Comparison | None = None
    reference: ExperimentConfig | None = None

    def __post_init__(self):
        roles = [model.role for model in self.models.values()]
        if roles.count("baseline") != 1:
            raise ValueError(f"Exactly one model must have role 'baseline', found {roles.count('baseline')}")
        calibrated = [name for name, model in self.models.items() if model.calibrated]
        if calibrated and self.calibration is None:
            raise ValueError(f"Models {calibrated} are calibrated but the config has no calibration section")
        allowed = CLASSIFIERS if self.task == "classification" else REGRESSORS
        wrong = [name for name, model in self.models.items() if model.estimator not in allowed]
        if wrong:
            raise ValueError(f"Models {wrong} do not fit a {self.task} task (allowed estimators: {allowed})")
        if self.task == "classification":
            classes = set((self.target.class_map or {}).values())
            if len(classes) != 2 or self.target.positive_class not in classes:
                raise ValueError("A classification config needs target.class_map with two classes, one of them "
                                 "being target.positive_class")
            if self.target.decision_threshold is None:
                raise ValueError("A classification config needs target.decision_threshold")
        else:
            if self.target.output is None or self.cv.stratify_bins is None:
                raise ValueError("A regression config needs target.output and cv.stratify_bins")
        if self.role == "sensitivity" and (self.comparison is None or self.reference is None):
            raise ValueError("A sensitivity config needs a comparison section with its reference_config")
        if self.role == "publishable" and self.comparison is not None:
            raise ValueError("Only a sensitivity config has a comparison section")
        numeric = set(self.features.numeric)
        for model in self.models.values():
            if model.spline is not None and not set(model.spline.columns) <= numeric:
                raise ValueError(f"models.{model.name}.spline uses non-numeric columns")
        if not set(self.support.columns) <= numeric:
            raise ValueError("support.columns must be numeric feature columns")
        if self.support.partition_column not in self.features.categorical:
            raise ValueError("support.partition_column must be a categorical feature column")
        if not set(self.interface.selector.columns.values()) <= numeric:
            raise ValueError("interface.selector.columns must point to numeric feature columns")

    @property
    def name(self) -> str:
        return self.experiment.name

    @property
    def version(self) -> str:
        return self.experiment.version

    @property
    def model_version(self) -> str:
        return f"{self.name}-{self.version}"

    @property
    def role(self) -> str:
        return self.experiment.role

    @property
    def task(self) -> str:
        return self.experiment.task

    @property
    def decisions(self) -> list[str]:
        return list(self.experiment.decisions)

    @property
    def baseline_name(self) -> str:
        return next(name for name, model in self.models.items() if model.role == "baseline")

    @property
    def candidate_names(self) -> list[str]:
        return [name for name, model in self.models.items() if model.role == "candidate"]

    @property
    def feature_columns(self) -> list[str]:
        return self.features.columns


# ---------------------------------------------------------------------------
# Building the dataclasses from the parsed YAML
# ---------------------------------------------------------------------------
def _exclusions(items: Any, where: str) -> tuple[Exclusion, ...]:
    if not isinstance(items, list):
        raise ValueError(f"{where} must be a list")
    return tuple(Exclusion(**_check_keys(item, f"{where} entry", ("name", "column", "rule"), ("value",)))
                 for item in items)


def _product(item: Any) -> Product:
    item = _check_keys(item, "features.products entry", ("name", "columns"))
    return Product(name=item["name"], columns=_strings(item["columns"], f"features.products {item['name']!r} columns"))


def _model(name: str, item: Any) -> Model:
    item = _check_keys(item, f"models.{name}", ("role", "estimator"), ("params", "calibrated", "spline"))
    spline = None
    if item.get("spline") is not None:
        raw = _check_keys(item["spline"], f"models.{name}.spline", ("columns", "n_knots", "degree"))
        spline = Spline(columns=_strings(raw["columns"], f"models.{name}.spline.columns"), n_knots=raw["n_knots"],
                        degree=raw["degree"])
    params = item.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"models.{name}.params must be a mapping")
    return Model(name=name, role=item["role"], estimator=item["estimator"], params=dict(params),
                 calibrated=item.get("calibrated", False), spline=spline)


SECTIONS_REQUIRED = ("experiment", "dataset", "target", "features", "models", "cv", "selection", "support",
                     "interface", "report")
SECTIONS_OPTIONAL = ("calibration", "evaluation", "comparison")


def experiment_from_values(values: dict[str, Any], path: Path, sha256: str,
                           reference: ExperimentConfig | None = None) -> ExperimentConfig:
    """Build an ExperimentConfig from the parsed (and, for a sensitivity config, merged) YAML values."""
    _check_keys(values, "config", SECTIONS_REQUIRED, SECTIONS_OPTIONAL)
    ex = _check_keys(values["experiment"], "experiment", ("name", "version", "role", "task", "decisions"))
    ds = _check_keys(values["dataset"], "dataset",
                     ("name", "accession", "path", "group_column", "row_key", "id_columns", "exclusions"),
                     ("extra_exclusions",))
    tg = _check_keys(values["target"], "target", ("column",),
                     ("positive_class", "decision_threshold", "class_map", "output", "prediction_bounds",
                      "scaled_outputs"))
    ft = _check_keys(values["features"], "features", ("categorical", "numeric", "boolean"), ("products",))
    cv = _check_keys(values["cv"], "cv", ("n_splits", "n_repeats", "shuffle", "seed"), ("stratify_bins",))
    sl = _check_keys(values["selection"], "selection", ("tie_breaker_metric", "tie_breaker_direction", "must_beat"))
    sp = _check_keys(values["support"], "support",
                     ("columns", "partition_column", "range_percentile", "nearest_k", "low_confidence_below_support",
                      "high_confidence_min_support"), ("high_confidence_min_probability",))
    it = _check_keys(values["interface"], "interface",
                     ("range_decimals", "input_order", "fixed_inputs", "composition", "selector", "direct_inputs"))
    rp = _check_keys(values["report"], "report", ("pressure_level_decimals", "intended_use", "limitations"),
                     ("reference_queries",))
    if not isinstance(values["models"], dict) or not values["models"]:
        raise ValueError("models must map each model name to its settings")

    composition = _check_keys(it["composition"], "interface.composition", ("columns", "max_sum"))
    selector = _check_keys(it["selector"], "interface.selector",
                           ("input", "value_input", "source_column", "empty_value", "unit", "columns"))
    calibration = None
    if values.get("calibration") is not None:
        cal = _check_keys(values["calibration"], "calibration", ("method", "n_splits"))
        calibration = Calibration(method=cal["method"], n_splits=cal["n_splits"])
    evaluation = None
    if values.get("evaluation") is not None:
        ev = _check_keys(values["evaluation"], "evaluation", (), ("region_columns", "region_round"))
        evaluation = EvaluationSettings(
            region_columns=_strings(ev.get("region_columns", []), "evaluation.region_columns"),
            region_round=dict(ev.get("region_round", {})))
    comparison = None
    if values.get("comparison") is not None:
        cp = _check_keys(values["comparison"], "comparison",
                         ("reference_config", "max_abs_roc_auc_delta", "min_spearman"))
        comparison = Comparison(reference_config=resolve_path(cp["reference_config"]),
                                max_abs_roc_auc_delta=cp["max_abs_roc_auc_delta"], min_spearman=cp["min_spearman"])
    bounds = tg.get("prediction_bounds")
    return ExperimentConfig(
        path=path,
        sha256=sha256,
        experiment=Experiment(name=ex["name"], version=ex["version"], role=ex["role"], task=ex["task"],
                              decisions=_strings(ex["decisions"], "experiment.decisions")),
        dataset=Dataset(name=ds["name"], accession=ds["accession"], path=resolve_path(ds["path"]),
                        group_column=ds["group_column"], row_key=ds["row_key"],
                        id_columns=_strings(ds["id_columns"], "dataset.id_columns"),
                        exclusions=_exclusions(ds["exclusions"], "dataset.exclusions"),
                        extra_exclusions=_exclusions(ds.get("extra_exclusions", []), "dataset.extra_exclusions")),
        target=Target(column=tg["column"], positive_class=tg.get("positive_class"),
                      decision_threshold=tg.get("decision_threshold"),
                      class_map=_string_map(tg["class_map"], "target.class_map") if "class_map" in tg else None,
                      output=tg.get("output"),
                      prediction_bounds=tuple(bounds) if bounds is not None else None,
                      scaled_outputs=tuple(ScaledOutput(**_check_keys(item, "target.scaled_outputs entry",
                                                                      ("name", "input")))
                                           for item in tg.get("scaled_outputs", []))),
        features=Features(categorical=_strings(ft["categorical"], "features.categorical"),
                          numeric=_strings(ft["numeric"], "features.numeric"),
                          boolean=_strings(ft["boolean"], "features.boolean"),
                          products=tuple(_product(raw) for raw in ft.get("products", []))),
        models={name: _model(name, item) for name, item in values["models"].items()},
        cv=CrossValidation(n_splits=cv["n_splits"], n_repeats=cv["n_repeats"], shuffle=cv["shuffle"], seed=cv["seed"],
                           stratify_bins=cv.get("stratify_bins")),
        selection=Selection(tie_breaker_metric=sl["tie_breaker_metric"],
                            tie_breaker_direction=sl["tie_breaker_direction"],
                            must_beat=tuple(MetricRule(**_check_keys(item, "selection.must_beat entry",
                                                                     ("metric", "direction")))
                                            for item in sl["must_beat"])),
        support=Support(columns=_strings(sp["columns"], "support.columns"), partition_column=sp["partition_column"],
                        range_percentile=sp["range_percentile"], nearest_k=sp["nearest_k"],
                        low_confidence_below_support=sp["low_confidence_below_support"],
                        high_confidence_min_support=sp["high_confidence_min_support"],
                        high_confidence_min_probability=sp.get("high_confidence_min_probability")),
        interface=Interface(
            range_decimals=it["range_decimals"],
            input_order=_strings(it["input_order"], "interface.input_order"),
            fixed_inputs=dict(it["fixed_inputs"]),
            composition=Composition(columns=_strings(composition["columns"], "interface.composition.columns"),
                                    max_sum=composition["max_sum"]),
            selector=Selector(input=selector["input"], value_input=selector["value_input"],
                              source_column=selector["source_column"], empty_value=selector["empty_value"],
                              unit=selector["unit"],
                              columns=_string_map(selector["columns"], "interface.selector.columns")),
            direct_inputs=tuple(DirectInput(**_check_keys(item, "interface.direct_inputs entry",
                                                          ("name", "column", "type"), ("unit",)))
                                for item in it["direct_inputs"])),
        report=Report(pressure_level_decimals=rp["pressure_level_decimals"], intended_use=rp["intended_use"],
                      limitations=_strings(rp["limitations"], "report.limitations"),
                      reference_queries=tuple(ReferenceQuery(**_check_keys(item, "report.reference_queries entry",
                                                                           ("name", "inputs")))
                                              for item in rp.get("reference_queries", []))),
        calibration=calibration,
        evaluation=evaluation,
        comparison=comparison,
        reference=reference,
    )


# ---------------------------------------------------------------------------
# Reading the files
# ---------------------------------------------------------------------------
def read_yaml(path: str | Path) -> tuple[dict[str, Any], str]:
    """Parsed values and sha256 of one YAML file. `\\r\\n` is read as `\\n`, so neither depends on the checkout."""
    content = resolve_path(path).read_bytes().replace(b"\r\n", b"\n")
    values = yaml.safe_load(content.decode("utf-8"))
    if not isinstance(values, dict):
        raise ValueError(f"{path} must contain a mapping of sections")
    return values, hashlib.sha256(content).hexdigest()


def merge_sections(reference: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Every section of the reference, with the keys of each overriding section replaced one level deep."""
    merged = copy.deepcopy(reference)
    for name, section in overrides.items():
        merged[name] = {**merged.get(name, {}), **copy.deepcopy(section)}
    return merged


def load_values(path: str | Path) -> tuple[dict[str, Any], str, ExperimentConfig | None]:
    """The values of one experiment file after inheritance, its sha256, and its reference experiment if any."""
    values, sha256 = read_yaml(path)
    reference_path = (values.get("comparison") or {}).get("reference_config")
    if reference_path is None:
        return values, sha256, None
    reference = load_experiment(reference_path)
    if reference.comparison is not None:
        raise ValueError(f"{path}: the reference config {reference_path} must not be a sensitivity config itself")
    reference_values, _ = read_yaml(reference_path)
    return merge_sections(reference_values, values), sha256, reference


def load_experiment(path: str | Path) -> ExperimentConfig:
    config_path = resolve_path(path)
    values, sha256, reference = load_values(config_path)
    return experiment_from_values(values, config_path, sha256, reference)


def init_training_settings(directory: Path = TRAINING_CONFIG_DIR) -> dict[str, ExperimentConfig]:
    """Load every *.yml in `directory`, keyed by experiment name."""
    settings: dict[str, ExperimentConfig] = {}
    for path in sorted(directory.glob("*.yml")):
        config = load_experiment(path)
        if config.name in settings:
            raise ValueError(f"Experiment name {config.name!r} is used by {settings[config.name].path.name} "
                             f"and {path.name}")
        settings[config.name] = config
    return settings


def get_experiment(name_or_path: str | Path) -> ExperimentConfig:
    """An experiment by name (e.g. "flex-outcome") or by the path of its YAML file."""
    key = str(name_or_path)
    if key in training_settings:
        return training_settings[key]
    path = resolve_path(key)
    for config in training_settings.values():
        if config.path == path:
            return config
    if path.suffix.lower() == ".toml":
        raise ValueError(f"{key}: the training configs are YAML now; use the experiment name or its .yml file")
    if not path.is_file():
        raise ValueError(f"Unknown experiment {key!r}; known experiments: {sorted(training_settings)}")
    return load_experiment(path)


training_settings = init_training_settings()
