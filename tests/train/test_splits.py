from __future__ import annotations

import numpy as np

from flame.training.evaluate import evaluation_splits
from flame.training.models import fit_model
from flame.training.splits import grouped_calibration_splits


def test_outer_folds_never_share_a_group(synthetic_data, outcome_config):
    groups = synthetic_data.groups.to_numpy()
    for split in evaluation_splits(synthetic_data, outcome_config):
        assert not set(groups[split.train]) & set(groups[split.test])


def test_every_row_is_tested_exactly_once_per_repeat(synthetic_data, outcome_config):
    cv = outcome_config.cv
    splits = evaluation_splits(synthetic_data, outcome_config)
    assert len(splits) == cv.n_splits * cv.n_repeats
    for repeat in range(cv.n_repeats):
        tested = np.concatenate([split.test for split in splits if split.repeat == repeat])
        assert sorted(tested) == list(range(len(synthetic_data.rows)))


def test_repeats_use_different_fold_assignments(synthetic_data, outcome_config):
    splits = evaluation_splits(synthetic_data, outcome_config)
    first = [tuple(split.test) for split in splits if split.repeat == 0]
    second = [tuple(split.test) for split in splits if split.repeat == 1]
    assert first != second


def test_splits_are_reproducible(synthetic_data, outcome_config):
    first = evaluation_splits(synthetic_data, outcome_config)
    second = evaluation_splits(synthetic_data, outcome_config)
    assert all(np.array_equal(a.test, b.test) and np.array_equal(a.train, b.train) for a, b in zip(first, second))


def test_calibration_splits_never_share_a_group(synthetic_data, outcome_config):
    groups = synthetic_data.groups.to_numpy()
    splits = grouped_calibration_splits(synthetic_data.target, synthetic_data.groups,
                                        outcome_config.calibration, seed=3)
    assert len(splits) == outcome_config.calibration.n_splits
    for fit_rows, calibration_rows in splits:
        assert not set(groups[fit_rows]) & set(groups[calibration_rows])


def test_fitted_calibrated_model_keeps_groups_apart_inside_each_outer_fold(synthetic_data, outcome_config):
    split = evaluation_splits(synthetic_data, outcome_config)[0]
    train_groups = synthetic_data.groups.iloc[split.train]
    pipeline = fit_model("hgb", outcome_config, synthetic_data.features.iloc[split.train],
                         synthetic_data.target.iloc[split.train], train_groups, split.seed)
    calibrated = pipeline.named_steps["estimator"]
    assert len(calibrated.calibrated_classifiers_) == outcome_config.calibration.n_splits
    for fit_rows, calibration_rows in calibrated.cv:
        fit_groups = set(train_groups.to_numpy()[fit_rows])
        calibration_groups = set(train_groups.to_numpy()[calibration_rows])
        assert not fit_groups & calibration_groups
        assert not calibration_groups & set(synthetic_data.groups.to_numpy()[split.test])
