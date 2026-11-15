from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

from flame.training import flex_check
from flame.training.config import ExperimentConfig
from flame.training.datasets import TrainingData
from flame.training.evaluate import Evaluation, absolute_error_by_region
from flame.training.flex_data import TRAIN_ROOT

DECIMALS = 4
DECISION_LOG = "reports/decision_log.md"


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(TRAIN_ROOT).as_posix()
    except ValueError:
        return path.name


def number(value: Any, decimals: int = DECIMALS) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{decimals}f}"


def mean_std(values: pd.Series) -> str:
    return f"{number(values.mean())} ± {number(values.std(ddof=0))}"


def markdown_table(frame: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def key_value_table(items: list[tuple[str, Any]], headers: tuple[str, str] = ("Item", "Value")) -> str:
    return markdown_table(pd.DataFrame(items, columns=list(headers)))


def counts_text(counts: dict[str, Any]) -> str:
    return ", ".join(f"{name} {value}" for name, value in counts.items())


def target_summary(data: TrainingData, config: ExperimentConfig) -> str:
    if config.task == "classification":
        return counts_text(data.target.value_counts().to_dict())
    stats = data.target.describe()
    return (f"{config.section('target')['column']}: min {number(stats['min'])}, mean {number(stats['mean'])}, "
            f"max {number(stats['max'])}, std {number(stats['std'])}")


def cv_text(config: ExperimentConfig) -> str:
    cv = config.section("cv")
    return (f"StratifiedGroupKFold grouped by `{config.section('dataset')['group_column']}`, {cv['n_splits']} folds "
            f"x {cv['n_repeats']} repeats, shuffle {str(cv['shuffle']).lower()}, seed {cv['seed']}")


def run_table(config: ExperimentConfig, data: TrainingData) -> str:
    dataset = config.section("dataset")
    return key_value_table([
        ("Config", f"`{display_path(config.path)}` (sha256 `{config.sha256}`)"),
        ("Data", f"`{dataset['name']}` ({dataset['accession']}), sha256 `{data.source_sha256}`"),
        ("Task", config.task),
        ("Rows used", f"{len(data.rows)} of {data.source_rows}"),
        ("Target", target_summary(data, config)),
        ("Groups", f"{data.groups.nunique()} (`{dataset['group_column']}`)"),
        ("Cross-validation", cv_text(config)),
        ("Decisions", f"{', '.join(config.decisions)} (see `{DECISION_LOG}`)"),
    ])


def exclusions_table(data: TrainingData) -> str:
    return key_value_table(list(data.exclusions.items()), ("Filter", "Rows removed"))


def metric_summary(metrics: pd.DataFrame, evaluation: Evaluation, config: ExperimentConfig) -> str:
    models = config.section("models")
    records = []
    for name, model in models.items():
        rows = metrics[metrics["model"] == name]
        records.append({"model": name, "role": model["role"],
                        **{metric: mean_std(rows[metric]) for metric in evaluation.metric_names}})
    return markdown_table(pd.DataFrame(records))


def comparison_table(evaluation: Evaluation) -> str:
    records = [{
        "candidate": item["candidate"],
        "metric": item["metric"],
        "must be": item["direction"],
        "baseline mean": number(item["baseline_mean"]),
        "candidate mean": number(item["candidate_mean"]),
        "mean difference": number(item["mean_difference"]),
        "std of difference": number(item["std_difference"]),
        "repeats better": f"{item['repeats_better']} of {item['repeats']}",
        "beats baseline": "yes" if item["beats_baseline"] else "no",
    } for item in evaluation.selection["comparisons"]]
    return markdown_table(pd.DataFrame(records))


def undefined_fold_note(evaluation: Evaluation) -> str:
    undefined = {metric: int(evaluation.fold_metrics[metric].isna().sum()) for metric in evaluation.metric_names}
    undefined = {metric: count for metric, count in undefined.items() if count}
    if not undefined:
        return "Every metric is defined on every test fold."
    return (f"Fold values that are undefined because the test fold lacks a class, and are left out of the mean: "
            f"{counts_text(undefined)}.")


def region_table(evaluation: Evaluation, data: TrainingData, config: ExperimentConfig) -> str:
    table = absolute_error_by_region(evaluation, data, config).reset_index()
    for model in config.section("models"):
        table[model] = table[model].map(number)
    return markdown_table(table)


def evaluation_report(config: ExperimentConfig, data: TrainingData, evaluation: Evaluation) -> str:
    folds = evaluation.fold_metrics.groupby("model").size().iloc[0]
    repeats = config.section("cv")["n_repeats"]
    parts = [
        f"# {config.name} {config.version}: evaluation",
        f"Generated by `python main.py train evaluate --config {display_path(config.path)}`. "
        "Every model is trained and tested on the same splits.",
        run_table(config, data),
        "## Rows excluded",
        exclusions_table(data),
        f"## Metrics per test fold (mean ± std over {folds} folds)",
        metric_summary(evaluation.fold_metrics, evaluation, config),
        undefined_fold_note(evaluation),
        f"## Metrics on pooled out-of-fold predictions (mean ± std over {repeats} repeats)",
        "Each repeat predicts every row exactly once; the metric is computed on all rows of the repeat.",
        metric_summary(evaluation.repeat_metrics, evaluation, config),
        "## Paired comparison with the baseline",
        f"Unit of comparison: {evaluation.selection['unit']}.",
        comparison_table(evaluation) if evaluation.selection["comparisons"] else "No candidates configured.",
        "## Selection",
        f"Selected model: **{evaluation.selection['selected']}**. {evaluation.selection['reason']}.",
    ]
    if config.task == "regression":
        parts += ["## Mean absolute error by region (all out-of-fold predictions)",
                  region_table(evaluation, data, config)]
    return "\n\n".join(parts) + "\n"


def write_evaluation_report(config: ExperimentConfig, data: TrainingData, evaluation: Evaluation,
                            reports_root: Path) -> Path:
    path = reports_root / f"{config.name}_evaluation.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(evaluation_report(config, data, evaluation), encoding="utf-8", newline="\n")
    return path


def inputs_table(info: dict[str, Any]) -> str:
    selector = info["input_mapping"]["selector"]
    records = []
    for name, field in info["inputs"].items():
        if field["type"] == "enum":
            tested = ", ".join(field["values"])
        elif name == selector["value_input"]:
            tested = "; ".join(f"{key}: {limits['min']:g} to {limits['max']:g}"
                               for key, limits in field["by_selector"].items())
        else:
            tested = f"{field['min']:g} to {field['max']:g}"
        records.append({"input": name, "type": field["type"], "tested range": tested, "unit": field.get("unit", "")})
    return markdown_table(pd.DataFrame(records))


def flex_coverage_facts(config: ExperimentConfig, data: TrainingData, source_table: pd.DataFrame) -> str:
    rows = data.rows
    selector = config.section("interface")["selector"]
    decimals = config.section("report")["pressure_level_decimals"]
    levels = rows["pressure_atm"].round(decimals).value_counts().sort_index()
    with_suppressant = rows[rows[selector["source_column"]] != selector["empty_value"]]
    moon_source, moon_rows = flex_check.in_moon_box(source_table), flex_check.in_moon_box(rows)
    moon = flex_check.MOON
    heptane_extinction = (rows["fuel"] == "heptane") & (rows["test_end"] == "extinction")
    return key_value_table([
        ("Pressure levels in the training rows", ", ".join(f"{level:g} atm: {count} rows"
                                                           for level, count in levels.items())),
        ("Highest O2 tested together with a suppressant",
         f"{with_suppressant['o2'].max():g}" if len(with_suppressant) else "n/a"),
        (f"Moon-like tests in the source table ({moon['p_min']:g} to {moon['p_max']:g} atm, O2 >= {moon['o2_min']:g})",
         f"{int(moon_source.sum())}, test ends: "
         f"{counts_text(source_table.loc[moon_source, 'test_end'].value_counts().to_dict())}"),
        ("Moon-like tests in the training rows",
         f"{int(moon_rows.sum())} in {rows.loc[moon_rows, 'atmosphere_id'].nunique()} atmosphere groups; with a "
         f"suppressant: {int((moon_rows & (rows[selector['source_column']] != selector['empty_value'])).sum())}"),
        ("Fiber-supported training rows", f"{int(rows['fiber'].astype(bool).sum())} of {len(rows)}"),
        ("Heptane extinction rows whose TestSheet comment mentions cool flames",
         f"{int((heptane_extinction & rows['mentions_cool_flame']).sum())} of {int(heptane_extinction.sum())}"),
        ("Original test end of the training rows", counts_text(rows["test_end"].value_counts().to_dict())),
    ], ("Fact", "Value"))


def algorithm_text(info: dict[str, Any]) -> str:
    algorithm = info["algorithm"]
    calibration = f", calibrated ({algorithm['calibration']['method']}, grouped)" if algorithm["calibrated"] else ""
    return f"`{algorithm['name']}` ({algorithm['estimator']}{calibration})"


def task_text(info: dict[str, Any]) -> str:
    if info["task"] == "classification":
        return (f"classification; classes {', '.join(info['class_labels'])}; positive class "
                f"{info['positive_class']}; decision threshold {info['decision_threshold']:g}")
    bounds = info.get("prediction_bounds")
    bounded = f"; predictions clipped to {bounds[0]:g}–{bounds[1]:g}" if bounds else ""
    return f"regression of `{info['prediction_output']}`{bounded}"


def support_text(info: dict[str, Any]) -> str:
    support = info["support"]
    rule = support["confidence"]
    probability = rule.get("high_confidence_min_probability")
    high = f"in range and support_n >= {rule['high_confidence_min_support']}"
    if probability is not None:
        high += f" and probability of the predicted class >= {probability:g}"
    thresholds = ", ".join(f"{name}: {number(value)}" for name, value in support["thresholds"].items())
    return "\n".join([
        f"- Space: {', '.join(support['columns'])}, standardized, one index per `{support['partition_column']}`.",
        f"- Joint range threshold (percentile {support['range_percentile']:g} of each training row's distance to "
        f"its nearest other test): {thresholds}.",
        "- `in_tested_range` is true when the nearest test is within the threshold and no input is outside its own "
        "tested range.",
        "- `support_n` counts training tests within the threshold; `nearest_tests` lists the "
        f"{support['nearest_k']} closest.",
        f"- `confidence`: low if out of range or support_n < {rule['low_confidence_below_support']}; high if {high}; "
        "otherwise medium.",
    ])


def prediction_text(result: dict[str, Any], info: dict[str, Any]) -> str:
    if info["task"] == "classification":
        return f"{result['outcome']} (p = {number(result['probabilities'][result['outcome']], 3)})"
    return ", ".join(f"{name} = {number(result[name], 3)}"
                     for name in [info["prediction_output"], *[s["name"] for s in info.get("scaled_outputs", [])]])


def reference_query_table(results: list[dict[str, Any]], info: dict[str, Any]) -> str:
    fields = info["support"]["nearest_test_fields"]
    records = [{
        "query": item["name"],
        "inputs": ", ".join(f"{key}={value}" for key, value in item["inputs"].items()),
        "prediction": prediction_text(item["result"], info),
        "in_tested_range": str(item["result"]["in_tested_range"]).lower(),
        "out_of_range_fields": ", ".join(item["result"]["out_of_range_fields"]) or "none",
        "support_n": item["result"]["support_n"],
        "confidence": item["result"]["confidence"],
        "nearest tests": ", ".join(str(test[fields[0]]) for test in item["result"]["nearest_tests"]),
    } for item in results]
    return markdown_table(pd.DataFrame(records))


def reproducibility_table(config: ExperimentConfig, info: dict[str, Any]) -> str:
    code = info["code_version"]
    code_text = f"`{code['commit']}`{' with uncommitted changes' if code['dirty'] else ''}" if code else "unknown"
    return key_value_table([
        ("Config", f"`{display_path(config.path)}` (sha256 `{info['config_sha256']}`)"),
        ("Data sha256", f"`{info['training_source']['sha256']}`"),
        ("Cross-validation", cv_text(config)),
        ("Library versions", ", ".join(f"{name} {version}" for name, version in info["versions"].items())),
        ("Code version", code_text),
        ("Decisions", f"{', '.join(info['decisions'])} (see `{DECISION_LOG}`)"),
        ("Rebuild", f"`python main.py train export --config {display_path(config.path)}`"),
    ])


def model_card(config: ExperimentConfig, data: TrainingData, evaluation: Evaluation, info: dict[str, Any],
               reference_results: list[dict[str, Any]], source_table: pd.DataFrame) -> str:
    report = config.section("report")
    constraints = "; ".join(f"{' + '.join(c['columns'])} <= {c['max']:g}" for c in info["constraints"])
    fixed = ", ".join(f"{name} = {str(value).lower()}" for name, value in info["fixed_inputs"].items())
    parts = [
        f"# Model card: {info['model_version']}",
        f"Generated by `python main.py train export --config {display_path(config.path)}`.",
        "## Intended use",
        report["intended_use"],
        f"Outputs: {', '.join(f'`{name}`' for name in info['outputs'])}.",
        "## Model",
        key_value_table([
            ("Model version", info["model_version"]),
            ("Selected algorithm", algorithm_text(info)),
            ("Task", task_text(info)),
            ("Artifact files", ", ".join(f"`{name}`" for name in info["files"].values())),
        ]),
        "## Training data",
        run_table(config, data),
        "Rows excluded:",
        exclusions_table(data),
        "## Inputs and tested ranges",
        inputs_table(info),
        f"Fixed inputs (not exposed): {fixed}. Constraint: {constraints}; the remainder of the mixture is N2.",
        "## Results (grouped cross-validation)",
        f"Pooled out-of-fold predictions, mean ± std over {config.section('cv')['n_repeats']} repeats. "
        f"Details: `reports/{config.name}_evaluation.md`.",
        metric_summary(evaluation.repeat_metrics, evaluation, config),
        f"Selected model: **{evaluation.selection['selected']}**. {evaluation.selection['reason']}.",
        "## Support and confidence",
        support_text(info),
        "## Coverage facts",
        flex_coverage_facts(config, data, source_table),
        "## Limitations",
        "\n".join(f"- {item}" for item in report["limitations"]),
        "## Reference queries",
        reference_query_table(reference_results, info) if reference_results else "No reference queries configured.",
        "## Reproducibility",
        reproducibility_table(config, info),
    ]
    return "\n\n".join(parts) + "\n"


def sensitivity_report(config: ExperimentConfig, result: dict[str, Any]) -> str:
    comparison = config.section("comparison")
    reference = config.reference
    verdict = "consistent" if result["consistent"] else "not consistent"
    parts = [
        f"# {reference.name} {reference.version}: target sensitivity check",
        f"Generated by `python main.py train compare-targets --config {display_path(config.path)}`.",
        "The reference target and the sensitivity target are compared with the same algorithm, on the same grouped "
        "splits, on the rows both targets share.",
        key_value_table([
            ("Reference config", f"`{display_path(reference.path)}` (sha256 `{reference.sha256}`)"),
            ("Sensitivity config", f"`{display_path(config.path)}` (sha256 `{config.sha256}`)"),
            ("Data sha256", f"`{result['source_sha256']}`"),
            ("Algorithm compared", f"`{result['model']}` (the model selected for the reference target)"),
            ("Reference rows", f"{result['reference_rows']} ({counts_text(result['reference_counts'])})"),
            ("Sensitivity rows", f"{result['sensitivity_rows']} ({counts_text(result['sensitivity_counts'])})"),
            ("Shared rows", result["shared_rows"]),
            ("Cross-validation", cv_text(reference)),
            ("Decisions", f"{', '.join(config.decisions)} (see `{DECISION_LOG}`)"),
        ]),
        "## Ranking quality on the shared test rows",
        f"ROC-AUC of the pooled out-of-fold predictions on the shared rows, mean ± std over "
        f"{reference.section('cv')['n_repeats']} repeats.",
        markdown_table(pd.DataFrame([
            {"target": "reference", "roc_auc": result["reference_auc_text"]},
            {"target": "sensitivity", "roc_auc": result["sensitivity_auc_text"]},
        ])),
        "## Criterion fixed before the run",
        "The correlation compares the two final models, each fitted on all of its rows, on the shared rows.",
        key_value_table([
            ("ROC-AUC difference (reference - sensitivity)", number(result["auc_delta"])),
            ("Allowed absolute difference", f"{comparison['max_abs_roc_auc_delta']:g}"),
            ("Spearman correlation of the predicted probabilities", number(result["spearman"])),
            ("Required correlation", f"{comparison['min_spearman']:g}"),
        ], ("Criterion", "Value")),
        "## Verdict",
        f"The two targets are **{verdict}** under the criterion fixed before the run (decision D-02).",
    ]
    return "\n\n".join(parts) + "\n"
