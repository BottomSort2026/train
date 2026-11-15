from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TRAIN_ROOT = Path(__file__).resolve().parents[2]
for path in (TRAIN_ROOT, TRAIN_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from configs.training.config_training import ExperimentConfig, load_experiment
from flame.training.datasets import TrainingData, build_training_data
from flame.training.export import export_from_data
from flame.training.inference import load_bundle

OUTCOME_CONFIG = "configs/training/flex_outcome.yml"
SENSITIVITY_CONFIG = "configs/training/flex_outcome_sensitivity.yml"
EXTINCTION_CONFIG = "configs/training/flex_extinction.yml"
SYNTHETIC_SHA256 = "0" * 64
FAST_REPEATS = 2
FAST_BOOSTING_ITERATIONS = 20
EXCLUDED_BY_FILTER = {"duplicate_test": 1, "missing_initial_diameter": 2, "high_pressure": 2, "unknown_fiber": 1}


def with_values(config: ExperimentConfig, **sections) -> ExperimentConfig:
    return replace(config, **sections)


def faster(config: ExperimentConfig) -> ExperimentConfig:
    models = {}
    for name, model in config.models.items():
        if "max_iter" in model.params and model.estimator.startswith("hist_gradient_boosting"):
            model = replace(model, params={**model.params, "max_iter": FAST_BOOSTING_ITERATIONS})
        models[name] = model
    reference = faster(config.reference) if config.reference is not None else None
    return replace(config, cv=replace(config.cv, n_repeats=FAST_REPEATS), models=models, reference=reference)


def synthetic_flex_table() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    records = []
    for pressure in (0.7, 1.0):
        for o2 in (0.14, 0.18, 0.21, 0.25, 0.30):
            for suppressant, fraction in (("none", 0.0), ("co2", 0.2), ("he", 0.3)):
                for fuel in ("methanol", "heptane"):
                    for d0 in (2.0, 3.0, 4.0):
                        score = 14 * (0.22 - o2) + 3 * fraction + 0.4 * (d0 - 3) + rng.normal(0, 0.6)
                        records.append({
                            "fuel": fuel,
                            "pressure_atm": pressure + rng.normal(0, 0.005),
                            "o2": o2,
                            "co2": fraction if suppressant == "co2" else 0.0,
                            "he": fraction if suppressant == "he" else 0.0,
                            "suppressant": suppressant,
                            "suppressant_fraction": fraction,
                            "d0_mm": d0 + rng.normal(0, 0.1),
                            "score": score,
                            "atmosphere_id": f"{pressure:.1f}atm_O2-{o2:.2f}_{suppressant}",
                        })
    table = pd.DataFrame(records)
    not_extinguished = np.where(np.arange(len(table)) % 2 == 0, "disruption", "completion")
    table["test_end"] = np.where(table["score"] > 0, "extinction", not_extinguished)
    table["dext_over_d0"] = np.where(table["test_end"] == "extinction", np.clip(0.5 + 0.2 * table["score"], 0.2, 1.0),
                                     np.nan)
    table["dext_mm"] = table["dext_over_d0"] * table["d0_mm"]
    table["fiber"] = pd.Series(np.arange(len(table)) % 5 == 0, dtype=object)
    table["high_pressure"] = False
    table["duplicate_of"] = pd.array([pd.NA] * len(table), dtype="Int64")
    table["mentions_cool_flame"] = (table["fuel"] == "heptane") & (table["test_end"] == "extinction")
    table.insert(0, "table_row", np.arange(1, len(table) + 1))
    table.insert(1, "cir_id", [f"S{row:03d}" for row in table["table_row"]])
    table.insert(2, "testsheet_id", pd.array([f"FLEX-{row:03d}" for row in table["table_row"]], dtype="string"))
    table.loc[0, "duplicate_of"] = 5
    table.loc[[1, 2], ["d0_mm", "dext_mm", "dext_over_d0"]] = np.nan
    table.loc[[3, 4], "high_pressure"] = True
    table.loc[5, "fiber"] = None
    return table.drop(columns="score")


@pytest.fixture(scope="session")
def outcome_config() -> ExperimentConfig:
    return faster(load_experiment(OUTCOME_CONFIG))


@pytest.fixture(scope="session")
def sensitivity_config() -> ExperimentConfig:
    return faster(load_experiment(SENSITIVITY_CONFIG))


@pytest.fixture(scope="session")
def synthetic_table() -> pd.DataFrame:
    return synthetic_flex_table()


@pytest.fixture(scope="session")
def synthetic_data(synthetic_table: pd.DataFrame, outcome_config: ExperimentConfig) -> TrainingData:
    return build_training_data(synthetic_table, outcome_config, SYNTHETIC_SHA256)


@pytest.fixture(scope="session")
def synthetic_export(synthetic_table: pd.DataFrame, synthetic_data: TrainingData, outcome_config: ExperimentConfig,
                     tmp_path_factory: pytest.TempPathFactory) -> dict:
    root = tmp_path_factory.mktemp("synthetic_export")
    result = export_from_data(outcome_config, synthetic_data, synthetic_table, root / "artifacts", root / "reports")
    return {**result, "bundle": load_bundle(result["artifact_dir"]), "root": root}


@pytest.fixture(scope="session")
def config_with():
    return with_values


@pytest.fixture()
def typical_inputs() -> dict:
    return {"fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "suppressant": "co2", "suppressant_fraction": 0.2,
            "droplet_d0_mm": 3.0}
