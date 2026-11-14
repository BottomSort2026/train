from __future__ import annotations

import pandas as pd
import pytest

from flame.training.evaluate import classification_metrics, regression_metrics, select_model

BASELINE = {"macro_f1": 0.80, "brier": 0.130}


def repeat_metrics(**models: dict) -> pd.DataFrame:
    records = []
    for name, metrics in models.items():
        for repeat in range(3):
            records.append({"model": name, "repeat": repeat, **metrics})
    return pd.DataFrame(records)


def test_candidate_better_on_both_metrics_is_selected(outcome_config):
    metrics = repeat_metrics(logreg=BASELINE, logreg_spline={"macro_f1": 0.82, "brier": 0.120},
                             hgb={"macro_f1": 0.70, "brier": 0.150})
    selection = select_model(metrics, outcome_config)
    assert selection["selected"] == "logreg_spline"
    assert selection["qualifying"] == ["logreg_spline"]


def test_candidate_better_on_one_metric_only_loses_to_the_baseline(outcome_config):
    metrics = repeat_metrics(logreg=BASELINE, logreg_spline={"macro_f1": 0.85, "brier": 0.131},
                             hgb={"macro_f1": 0.79, "brier": 0.100})
    selection = select_model(metrics, outcome_config)
    assert selection["selected"] == "logreg"
    assert selection["qualifying"] == []


def test_candidate_that_ties_the_baseline_does_not_replace_it(outcome_config):
    metrics = repeat_metrics(logreg=BASELINE, logreg_spline=BASELINE, hgb=BASELINE)
    selection = select_model(metrics, outcome_config)
    assert selection["selected"] == "logreg"
    assert not any(item["beats_baseline"] for item in selection["comparisons"])


def test_two_qualifying_candidates_are_ranked_by_lowest_brier(outcome_config):
    metrics = repeat_metrics(logreg=BASELINE, logreg_spline={"macro_f1": 0.90, "brier": 0.120},
                             hgb={"macro_f1": 0.81, "brier": 0.110})
    selection = select_model(metrics, outcome_config)
    assert selection["qualifying"] == ["logreg_spline", "hgb"]
    assert selection["selected"] == "hgb"


def test_comparison_is_paired_by_repeat(outcome_config):
    metrics = pd.DataFrame([
        {"model": "logreg", "repeat": 0, "macro_f1": 0.70, "brier": 0.20},
        {"model": "logreg", "repeat": 1, "macro_f1": 0.90, "brier": 0.10},
        {"model": "logreg_spline", "repeat": 0, "macro_f1": 0.72, "brier": 0.19},
        {"model": "logreg_spline", "repeat": 1, "macro_f1": 0.92, "brier": 0.09},
        {"model": "hgb", "repeat": 0, "macro_f1": 0.60, "brier": 0.30},
        {"model": "hgb", "repeat": 1, "macro_f1": 0.60, "brier": 0.30},
    ])
    selection = select_model(metrics, outcome_config)
    spline_f1 = next(item for item in selection["comparisons"]
                     if item["candidate"] == "logreg_spline" and item["metric"] == "macro_f1")
    assert spline_f1["mean_difference"] == pytest.approx(0.02)
    assert spline_f1["std_difference"] == pytest.approx(0.0)
    assert spline_f1["repeats_better"] == 2
    assert selection["selected"] == "logreg_spline"


def test_regression_rule_needs_a_strictly_lower_error(outcome_config, config_with):
    config = config_with(outcome_config, selection={
        "tie_breaker_metric": "mae", "tie_breaker_direction": "lower",
        "must_beat": [{"metric": "mae", "direction": "lower"}]})
    better = repeat_metrics(logreg={"mae": 0.10}, logreg_spline={"mae": 0.09}, hgb={"mae": 0.08})
    equal = repeat_metrics(logreg={"mae": 0.10}, logreg_spline={"mae": 0.10}, hgb={"mae": 0.11})
    assert select_model(better, config)["selected"] == "hgb"
    assert select_model(equal, config)["selected"] == "logreg"


def test_classification_metrics_follow_the_decision_threshold(outcome_config):
    target = outcome_config.section("target")
    truth = pd.Series(["extinction", "extinction", "no_extinction", "no_extinction"])
    metrics = classification_metrics(truth, pd.Series([0.9, 0.5, 0.4, 0.6]).to_numpy(), target)
    assert metrics["recall_extinction"] == pytest.approx(1.0)
    assert metrics["recall_no_extinction"] == pytest.approx(0.5)
    assert metrics["brier"] == pytest.approx((0.01 + 0.25 + 0.16 + 0.36) / 4)
    assert metrics["roc_auc"] == pytest.approx(0.75)


def test_metrics_of_a_single_class_fold_are_marked_undefined(outcome_config):
    target = outcome_config.section("target")
    truth = pd.Series(["extinction", "extinction"])
    metrics = classification_metrics(truth, pd.Series([0.9, 0.2]).to_numpy(), target)
    assert pd.isna(metrics["roc_auc"])
    assert pd.isna(metrics["recall_no_extinction"])
    assert metrics["recall_extinction"] == pytest.approx(0.5)


def test_regression_metrics_match_hand_computed_values():
    metrics = regression_metrics(pd.Series([0.2, 0.4, 0.6]), pd.Series([0.3, 0.4, 0.5]).to_numpy())
    assert metrics["mae"] == pytest.approx(0.2 / 3)
    assert metrics["rmse"] == pytest.approx((0.02 / 3) ** 0.5)
    assert metrics["r2"] == pytest.approx(0.75)
