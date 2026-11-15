from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from configs.training.config_training import ESTIMATOR_NAMES, Calibration, ExperimentConfig, Features, Model
from flame.training.features import build_preprocessor
from flame.training.splits import grouped_calibration_splits

ESTIMATORS: dict[str, type] = {
    "logistic_regression": LogisticRegression,
    "hist_gradient_boosting_classifier": HistGradientBoostingClassifier,
    "ridge": Ridge,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
}


if set(ESTIMATORS) != set(ESTIMATOR_NAMES):
    raise RuntimeError("ESTIMATORS and configs.training.config_training.ESTIMATOR_NAMES list different estimators")


def is_calibrated(model: Model) -> bool:
    return model.calibrated


def build_model(model: Model, features: Features, calibration: Calibration | None = None,
                calibration_splits: list[tuple[np.ndarray, np.ndarray]] | None = None) -> Pipeline:
    if model.estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {model.estimator!r}; known: {sorted(ESTIMATORS)}")
    estimator = ESTIMATORS[model.estimator](**model.params)
    if is_calibrated(model):
        if calibration is None or calibration_splits is None:
            raise ValueError("A calibrated model needs the calibration settings and grouped calibration splits")
        estimator = CalibratedClassifierCV(estimator, method=calibration.method, cv=calibration_splits)
    return Pipeline([("preprocess", build_preprocessor(features, model.spline)), ("estimator", estimator)])


def fit_model(model_name: str, config: ExperimentConfig, features: pd.DataFrame, target: pd.Series,
              groups: pd.Series, seed: int) -> Pipeline:
    model = config.models[model_name]
    calibration = config.calibration
    calibration_splits = None
    if is_calibrated(model):
        calibration_splits = grouped_calibration_splits(target, groups, calibration, seed)
    pipeline = build_model(model, config.features, calibration, calibration_splits)
    return pipeline.fit(features, target)


def predict_values(pipeline: Pipeline, features: pd.DataFrame, config: ExperimentConfig) -> np.ndarray:
    target = config.target
    if config.task == "classification":
        positive_index = list(pipeline.classes_).index(target.positive_class)
        return pipeline.predict_proba(features)[:, positive_index]
    predictions = pipeline.predict(features)
    bounds = target.prediction_bounds
    return np.clip(predictions, bounds[0], bounds[1]) if bounds else predictions
