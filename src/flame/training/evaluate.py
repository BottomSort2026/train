from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score, roc_auc_score

from flame.training.config import ExperimentConfig
from flame.training.datasets import TrainingData
from flame.training.models import fit_model, predict_values
from flame.training.splits import Split, repeated_group_splits, stratification_labels

SELECTION_UNIT = "pooled out-of-fold predictions of each repeat"


@dataclass(frozen=True)
class Evaluation:
    fold_metrics: pd.DataFrame
    repeat_metrics: pd.DataFrame
    predictions: pd.DataFrame
    selection: dict[str, Any]
    metric_names: list[str]


def class_labels(target: dict[str, Any]) -> tuple[str, str]:
    positive = target["positive_class"]
    negative = next(label for label in sorted(set(target["class_map"].values())) if label != positive)
    return positive, negative


def predicted_labels(positive_probability: np.ndarray, target: dict[str, Any]) -> np.ndarray:
    positive, negative = class_labels(target)
    return np.where(positive_probability >= target["decision_threshold"], positive, negative)


def class_recall(truth: np.ndarray, predicted: np.ndarray, label: str) -> float:
    members = truth == label
    return float((predicted[members] == label).mean()) if members.any() else float("nan")


def classification_metrics(truth: pd.Series, positive_probability: np.ndarray, target: dict[str, Any]) -> dict:
    positive, negative = class_labels(target)
    observed = truth.to_numpy()
    predicted = predicted_labels(positive_probability, target)
    is_positive = (observed == positive).astype(float)
    both_classes = 0.0 < is_positive.mean() < 1.0
    return {
        "macro_f1": float(f1_score(observed, predicted, labels=[positive, negative], average="macro",
                                   zero_division=0)),
        f"recall_{positive}": class_recall(observed, predicted, positive),
        f"recall_{negative}": class_recall(observed, predicted, negative),
        "brier": float(np.mean((positive_probability - is_positive) ** 2)),
        "roc_auc": float(roc_auc_score(is_positive, positive_probability)) if both_classes else float("nan"),
    }


def regression_metrics(truth: pd.Series, predicted: np.ndarray) -> dict:
    observed = truth.to_numpy()
    varies = len(observed) > 1 and float(np.ptp(observed)) > 0.0
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(np.mean((observed - predicted) ** 2))),
        "r2": float(r2_score(observed, predicted)) if varies else float("nan"),
    }


def score_predictions(truth: pd.Series, predictions: np.ndarray, config: ExperimentConfig) -> dict:
    if config.task == "classification":
        return classification_metrics(truth, predictions, config.section("target"))
    return regression_metrics(truth, predictions)


def evaluation_splits(data: TrainingData, config: ExperimentConfig) -> list[Split]:
    return repeated_group_splits(stratification_labels(data.target, config), data.groups, config.section("cv"))


def pooled_repeat_metrics(predictions: pd.DataFrame, target: pd.Series, config: ExperimentConfig) -> pd.DataFrame:
    records = []
    for (model, repeat), group in predictions.groupby(["model", "repeat"], sort=False):
        truth = target.iloc[group["row"].to_numpy()]
        records.append({"model": model, "repeat": repeat, "n_rows": len(group),
                        **score_predictions(truth, group["prediction"].to_numpy(), config)})
    return pd.DataFrame(records)


def paired_comparison(repeat_metrics: pd.DataFrame, baseline: str, candidate: str, metric: str,
                      direction: str) -> dict[str, Any]:
    by_repeat = repeat_metrics.pivot(index="repeat", columns="model", values=metric)
    difference = by_repeat[candidate] - by_repeat[baseline]
    mean_difference = float(difference.mean())
    better = difference > 0 if direction == "higher" else difference < 0
    return {
        "candidate": candidate,
        "metric": metric,
        "direction": direction,
        "baseline_mean": float(by_repeat[baseline].mean()),
        "candidate_mean": float(by_repeat[candidate].mean()),
        "mean_difference": mean_difference,
        "std_difference": float(difference.std(ddof=0)),
        "repeats_better": int(better.sum()),
        "repeats": int(len(difference)),
        "beats_baseline": bool(mean_difference > 0 if direction == "higher" else mean_difference < 0),
    }


def select_model(repeat_metrics: pd.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    selection = config.section("selection")
    baseline = config.baseline_name
    comparisons = [
        paired_comparison(repeat_metrics, baseline, candidate, rule["metric"], rule["direction"])
        for candidate in config.candidate_names
        for rule in selection["must_beat"]
    ]
    qualifying = [
        candidate for candidate in config.candidate_names
        if all(item["beats_baseline"] for item in comparisons if item["candidate"] == candidate)
    ]
    if qualifying:
        means = repeat_metrics.groupby("model")[selection["tie_breaker_metric"]].mean()
        ordered = means[qualifying].sort_values(ascending=selection["tie_breaker_direction"] == "lower")
        selected = str(ordered.index[0])
        reason = (f"{selected} beats the baseline {baseline} on every required metric"
                  + (f" and has the best {selection['tie_breaker_metric']} among {qualifying}"
                     if len(qualifying) > 1 else ""))
    else:
        selected = baseline
        reason = f"No candidate beats the baseline {baseline} on every required metric, so the baseline ships"
    return {"baseline": baseline, "selected": selected, "qualifying": qualifying, "unit": SELECTION_UNIT,
            "comparisons": comparisons, "reason": reason}


def evaluate_models(data: TrainingData, config: ExperimentConfig) -> Evaluation:
    fold_records = []
    prediction_frames = []
    for split in evaluation_splits(data, config):
        train_features, train_target = data.features.iloc[split.train], data.target.iloc[split.train]
        train_groups = data.groups.iloc[split.train]
        test_features, test_target = data.features.iloc[split.test], data.target.iloc[split.test]
        for model_name in config.section("models"):
            pipeline = fit_model(model_name, config, train_features, train_target, train_groups, split.seed)
            predictions = predict_values(pipeline, test_features, config)
            fold_records.append({"model": model_name, "repeat": split.repeat, "fold": split.fold,
                                 "n_test": len(split.test), **score_predictions(test_target, predictions, config)})
            prediction_frames.append(pd.DataFrame({"model": model_name, "repeat": split.repeat, "fold": split.fold,
                                                   "row": split.test, "prediction": predictions}))
    fold_metrics = pd.DataFrame(fold_records)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    repeat_metrics = pooled_repeat_metrics(predictions, data.target, config)
    metric_names = [c for c in fold_metrics.columns if c not in {"model", "repeat", "fold", "n_test"}]
    return Evaluation(fold_metrics=fold_metrics, repeat_metrics=repeat_metrics, predictions=predictions,
                      selection=select_model(repeat_metrics, config), metric_names=metric_names)


def region_labels(data: TrainingData, config: ExperimentConfig) -> pd.Series:
    regions = config.values.get("evaluation", {})
    parts = [data.rows[column].round(decimals).map(lambda value, name=column: f"{name}={value:g}")
             for column, decimals in regions.get("region_round", {}).items()]
    parts += [data.rows[column].astype(str).map(lambda value, name=column: f"{name}={value}")
              for column in regions.get("region_columns", [])]
    if not parts:
        return pd.Series("all", index=data.rows.index)
    return pd.concat(parts, axis=1).agg(", ".join, axis=1)


def absolute_error_by_region(evaluation: Evaluation, data: TrainingData, config: ExperimentConfig) -> pd.DataFrame:
    predictions = evaluation.predictions.copy()
    rows = predictions["row"].to_numpy()
    predictions["region"] = region_labels(data, config).to_numpy()[rows]
    predictions["absolute_error"] = (predictions["prediction"] - data.target.to_numpy()[rows]).abs()
    table = predictions.pivot_table(index="region", columns="model", values="absolute_error", aggfunc="mean")
    table.insert(0, "rows", predictions.groupby("region")["row"].nunique())
    return table[["rows", *config.section("models")]]
