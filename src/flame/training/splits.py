from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from configs.training.config_training import Calibration, CrossValidation, ExperimentConfig


@dataclass(frozen=True)
class Split:
    repeat: int
    fold: int
    seed: int
    train: np.ndarray
    test: np.ndarray


def stratification_labels(target: pd.Series, config: ExperimentConfig) -> pd.Series:
    if config.task == "classification":
        return target
    bins = config.cv.stratify_bins
    return pd.qcut(target, q=bins, labels=False, duplicates="drop")


def repeated_group_splits(labels: pd.Series, groups: pd.Series, cv: CrossValidation) -> list[Split]:
    placeholder = np.zeros(len(labels))
    splits: list[Split] = []
    for repeat in range(cv.n_repeats):
        seed = cv.seed + repeat
        splitter = StratifiedGroupKFold(n_splits=cv.n_splits, shuffle=cv.shuffle,
                                        random_state=seed if cv.shuffle else None)
        for fold, (train, test) in enumerate(splitter.split(placeholder, labels, groups)):
            splits.append(Split(repeat=repeat, fold=fold, seed=seed, train=train, test=test))
    return splits


def grouped_calibration_splits(target: pd.Series, groups: pd.Series, calibration: Calibration,
                               seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=calibration.n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(target)), target, groups))
    classes = set(target.unique())
    for fit_rows, calibration_rows in splits:
        if set(target.iloc[fit_rows]) != classes or set(target.iloc[calibration_rows]) != classes:
            raise ValueError("A grouped calibration split does not contain every class on both sides")
    return splits
