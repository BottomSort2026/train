from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from flame.training.config import ExperimentConfig
from flame.training.datasets import TrainingData, build_training_data, load_source_table
from flame.training.evaluate import evaluate_models, evaluation_splits
from flame.training.export import REPORTS_ROOT
from flame.training.models import fit_model, predict_values
from flame.training.reporting import mean_std, sensitivity_report


def reference_positions(reference_data: TrainingData, sensitivity_data: TrainingData) -> np.ndarray:
    reference_keys = reference_data.row_keys.to_numpy()
    sensitivity_keys = sensitivity_data.row_keys.to_numpy()
    if not np.isin(sensitivity_keys, reference_keys).all():
        raise ValueError("The sensitivity rows must be a subset of the reference rows")
    position_of_key = pd.Series(np.arange(len(reference_keys)), index=reference_keys)
    return position_of_key[sensitivity_keys].to_numpy()


def spearman_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_ranks = pd.Series(first).rank().to_numpy()
    second_ranks = pd.Series(second).rank().to_numpy()
    return float(np.corrcoef(first_ranks, second_ranks)[0, 1])


def sensitivity_out_of_fold(reference_data: TrainingData, sensitivity_data: TrainingData,
                            config: ExperimentConfig, model_name: str, to_reference: np.ndarray) -> pd.DataFrame:
    frames = []
    for split in evaluation_splits(reference_data, config.reference):
        in_train = np.isin(to_reference, split.train)
        in_test = np.isin(to_reference, split.test)
        if not in_test.any():
            continue
        pipeline = fit_model(model_name, config, sensitivity_data.features[in_train],
                             sensitivity_data.target[in_train], sensitivity_data.groups[in_train], split.seed)
        predictions = predict_values(pipeline, sensitivity_data.features[in_test], config)
        frames.append(pd.DataFrame({"repeat": split.repeat, "row": np.flatnonzero(in_test),
                                    "prediction": predictions}))
    return pd.concat(frames, ignore_index=True)


def roc_auc_per_repeat(predictions: pd.DataFrame, is_positive: np.ndarray) -> pd.Series:
    return pd.Series({
        repeat: float(roc_auc_score(is_positive[group["row"].to_numpy()], group["prediction"].to_numpy()))
        for repeat, group in predictions.groupby("repeat")
    })


def compare_targets_from_data(config: ExperimentConfig, reference_data: TrainingData,
                              sensitivity_data: TrainingData) -> dict[str, Any]:
    reference = config.reference
    comparison = config.section("comparison")
    positive = config.section("target")["positive_class"]
    to_reference = reference_positions(reference_data, sensitivity_data)
    is_positive = (sensitivity_data.target.to_numpy() == positive).astype(float)

    evaluation = evaluate_models(reference_data, reference)
    model_name = evaluation.selection["selected"]
    selected = evaluation.predictions[evaluation.predictions["model"] == model_name]
    shared_position = pd.Series(np.arange(len(to_reference)), index=to_reference)
    reference_shared = selected[selected["row"].isin(to_reference)].copy()
    reference_shared["row"] = shared_position[reference_shared["row"].to_numpy()].to_numpy()
    sensitivity_shared = sensitivity_out_of_fold(reference_data, sensitivity_data, config, model_name, to_reference)

    reference_auc = roc_auc_per_repeat(reference_shared, is_positive)
    sensitivity_auc = roc_auc_per_repeat(sensitivity_shared, is_positive)
    auc_delta = float(reference_auc.mean() - sensitivity_auc.mean())

    seed = reference.section("cv")["seed"]
    reference_model = fit_model(model_name, reference, reference_data.features, reference_data.target,
                                reference_data.groups, seed)
    sensitivity_model = fit_model(model_name, config, sensitivity_data.features, sensitivity_data.target,
                                  sensitivity_data.groups, seed)
    spearman = spearman_correlation(predict_values(reference_model, sensitivity_data.features, reference),
                                    predict_values(sensitivity_model, sensitivity_data.features, config))
    return {
        "model": model_name,
        "source_sha256": reference_data.source_sha256,
        "reference_rows": int(len(reference_data.rows)),
        "reference_counts": {name: int(count) for name, count in reference_data.target.value_counts().items()},
        "sensitivity_rows": int(len(sensitivity_data.rows)),
        "sensitivity_counts": {name: int(count) for name, count in sensitivity_data.target.value_counts().items()},
        "shared_rows": int(len(to_reference)),
        "reference_auc": float(reference_auc.mean()),
        "sensitivity_auc": float(sensitivity_auc.mean()),
        "reference_auc_text": mean_std(reference_auc),
        "sensitivity_auc_text": mean_std(sensitivity_auc),
        "auc_delta": auc_delta,
        "spearman": spearman,
        "consistent": bool(abs(auc_delta) <= comparison["max_abs_roc_auc_delta"]
                           and spearman >= comparison["min_spearman"]),
    }


def compare_targets(config: ExperimentConfig, reports_root: Path = REPORTS_ROOT) -> dict[str, Any]:
    if config.role != "sensitivity" or config.reference is None:
        raise ValueError(f"{config.name} is not a sensitivity config with a reference config")
    source_table, source_sha256 = load_source_table(config.reference)
    reference_data = build_training_data(source_table, config.reference, source_sha256)
    sensitivity_data = build_training_data(source_table, config, source_sha256)
    result = compare_targets_from_data(config, reference_data, sensitivity_data)
    path = reports_root / f"{config.reference.name}_target_sensitivity.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sensitivity_report(config, result), encoding="utf-8", newline="\n")
    return {**result, "report": path}
