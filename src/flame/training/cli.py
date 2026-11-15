from __future__ import annotations

import argparse

DEFAULT_SENSITIVITY_CONFIG = "flex-outcome-sensitivity"
CONFIG_HELP = "Experiment name (e.g. flex-outcome) or YAML file (e.g. configs/training/flex_outcome.yml)"


def _fetch_flex(args: argparse.Namespace) -> int:
    from flame.training import flex_data
    flex_data.download_raw(force=args.force)
    return 0


def _prepare_flex(args: argparse.Namespace) -> int:
    from flame.training import flex_data
    manifest = flex_data.prepare_flex()
    print(f"Wrote {flex_data.PROCESSED_CSV} ({manifest['rows']} rows, sha256 {manifest['sha256'][:12]}…)")
    for line in manifest["cleaning_log"]:
        print(f"  - {line}")
    return 0


def _check_flex(args: argparse.Namespace) -> int:
    from flame.training import flex_check
    print(f"Wrote {flex_check.run_check()}")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    from configs.training.config_training import get_experiment
    from flame.training.datasets import load_training_data
    from flame.training.evaluate import evaluate_models
    from flame.training.export import REPORTS_ROOT
    from flame.training.reporting import write_evaluation_report
    config = get_experiment(args.config)
    if config.role != "publishable":
        raise SystemExit(f"{config.name} is a {config.role} config; run it with: train compare-targets")
    data = load_training_data(config)
    evaluation = evaluate_models(data, config)
    print(f"Rows: {len(data.rows)} of {data.source_rows}; exclusions: {data.exclusions}")
    print(f"Selected model: {evaluation.selection['selected']}. {evaluation.selection['reason']}.")
    print(f"Wrote {write_evaluation_report(config, data, evaluation, REPORTS_ROOT)}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from configs.training.config_training import get_experiment
    from flame.training.export import export_experiment
    config = get_experiment(args.config)
    if config.role != "publishable":
        raise SystemExit(f"{config.name} is a {config.role} config and is never exported")
    result = export_experiment(config)
    print(f"Selected model: {result['selected']}")
    for key in ("artifact_dir", "model_card", "evaluation_report"):
        print(f"Wrote {result[key]}")
    return 0


def _compare_targets(args: argparse.Namespace) -> int:
    from configs.training.config_training import get_experiment
    from flame.training.sensitivity import compare_targets
    result = compare_targets(get_experiment(args.config))
    verdict = "consistent" if result["consistent"] else "not consistent"
    print(f"ROC-AUC difference {result['auc_delta']:.4f}, Spearman {result['spearman']:.4f}: {verdict}")
    print(f"Wrote {result['report']}")
    return 0


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    fetch = subparsers.add_parser("fetch-flex", help="Download the FLEX table and TestSheets (~0.3 MB) from PSI")
    fetch.add_argument("--force", action="store_true", help="Download again even if the files exist")
    fetch.set_defaults(func=_fetch_flex)

    prepare = subparsers.add_parser("prepare-flex", help="Clean FLEX into data/processed/flex_tests.csv + manifest")
    prepare.set_defaults(func=_prepare_flex)

    check = subparsers.add_parser("check-flex", help="Write the FLEX data check to reports/flex_data_check.md")
    check.set_defaults(func=_check_flex)

    evaluate = subparsers.add_parser("evaluate", help="Grouped cross-validation of the baseline and candidates")
    evaluate.add_argument("--config", required=True, help=CONFIG_HELP)
    evaluate.set_defaults(func=_evaluate)

    export = subparsers.add_parser("export", help="Evaluate, select, refit on all rows and write the artifact")
    export.add_argument("--config", required=True, help=CONFIG_HELP)
    export.set_defaults(func=_export)

    compare = subparsers.add_parser("compare-targets", help="Sensitivity check of the target definition")
    compare.add_argument("--config", default=DEFAULT_SENSITIVITY_CONFIG, help=f"Sensitivity experiment. {CONFIG_HELP}")
    compare.set_defaults(func=_compare_targets)
