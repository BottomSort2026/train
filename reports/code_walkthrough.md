# Code walkthrough: FLEX training pipeline

This document explains the training code in `train/`: what each module does, its public functions, why it
is built this way, how it is tested and how to run it. The source files carry no comments, so this is the
place where the code is explained. Decision identifiers (`D-xx`, `T-xx`, `O-xx`) refer to
`reports/decision_log.md`.

## 1. What the pipeline produces

The pipeline turns the published FLEX table into two models that the ML service can load:

| Model | Question it answers | Published algorithm |
|---|---|---|
| `flex-outcome-0.1` | Does the visible flame of a burning droplet go out on its own in this atmosphere? | Logistic regression (the baseline) |
| `flex-extinction-0.2` | If it goes out, how large is the droplet at that moment, relative to its initial size? | Shallow gradient boosting |

Each model is a folder in `artifacts/models/<model>-<version>/`:

| File | Content |
|---|---|
| `model.joblib` | The fitted scikit-learn pipeline: preprocessing plus estimator |
| `support.joblib` | The nearest-neighbor index that tells how much real data supports a prediction |
| `support_reference.csv` | One line per training test (IDs and inputs), in the row order the index uses |
| `model_info.json` | Inputs with tested ranges, outputs, data hash, cross-validation summary, decisions, versions |

and three generated reports in `reports/`: an evaluation report, a model card and, for `flex-outcome`, a
target sensitivity report.

## 2. Data flow

```
PSI-69 (public repository)
   |  train fetch-flex            flex_data.download_raw
   v
data/raw/PSI-69/                  experimental table + TestSheets
   |  train prepare-flex          flex_data.prepare_flex
   v
data/processed/flex_tests.csv     274 rows, flags instead of deleted rows; sha256 in flex_tests.json
   |  train check-flex            flex_check.run_check  ->  reports/flex_data_check.md
   |
   |  train evaluate / export     config.load_config  ->  datasets.load_training_data
   v
TrainingData (features, target, groups, rows)
   |  evaluate.evaluate_models    splits  ->  models.fit_model  ->  metrics  ->  select_model
   v
Evaluation (metrics per fold and per repeat, out-of-fold predictions, selection)
   |  export.export_from_data     refit on all rows, support.fit_support, build_model_info
   v
artifacts/models/<model>-<version>/   +   reports/<model>_evaluation.md   +   reports/<model>-<version>_model_card.md
   |  inference.reference_predict (the contract ml/ will reimplement)
   v
response: prediction, in_tested_range, out_of_range_fields, support_n, nearest_tests, confidence, ...
```

## 3. Configuration files

Every experiment is one TOML file in `configs/training/`. The code holds no modeling constants: seeds,
hyperparameters, thresholds, filters and report texts are all in the config (D-06, D-07, T-02).

| File | Role | Content |
|---|---|---|
| `flex_outcome.toml` | publishable | Binary outcome model (D-01), three models (D-06), support rule (D-07) |
| `flex_outcome_sensitivity.toml` | sensitivity | Only the differences from `flex_outcome.toml`: one extra exclusion and another class map (D-02, T-02) |
| `flex_extinction.toml` | publishable | Regression of the extinction ratio (D-08) |

Sections of a config:

| Section | Meaning |
|---|---|
| `[experiment]` | Name, version, role, task (`classification` or `regression`), decision IDs |
| `[dataset]` | Processed table, group column, row key, ID columns, and the ordered `[[dataset.exclusions]]` (D-03) |
| `[target]` | Target column; class map, positive class and decision threshold, or output name and bounds |
| `[features]` | Categorical, numeric and boolean pipeline inputs, and derived products such as `po2` (D-04) |
| `[models.<name>]` | Role (`baseline` or `candidate`), estimator, fixed parameters, optional spline and calibration flags |
| `[cv]`, `[calibration]` | Folds, repeats, shuffle, seed, strata for regression; calibration method and folds (D-05) |
| `[selection]` | Metrics a candidate must beat and the tie breaker (D-06, D-08) |
| `[support]` | Columns, partition column, percentile, neighbors, confidence thresholds (D-07) |
| `[interface]` | Exposed inputs, fixed inputs, mixture constraint, suppressant selector, rounding of ranges (D-04, T-11) |
| `[evaluation]` | Regions for the error table of a regression |
| `[report]` | Intended use, limitations and reference queries printed in the model card (T-15) |

## 4. Modules

All modules are in `src/flame/training/`.

### 4.1 `flex_data.py`: download, clean, load

| Function | Input | Output |
|---|---|---|
| `download_raw(force)` | — | Writes the table and the TestSheets to `data/raw/PSI-69/` |
| `read_raw_table(path)` | Raw CSV | The table as text, columns renamed |
| `read_testsheets(directory)` | TestSheet folder | One row per sheet: IDs, fiber flag, temperature, comment |
| `clean_flex(table, sheets)` | Both raw frames | The clean table and the list of changes made |
| `prepare_flex()` | — | Writes `flex_tests.csv` and its manifest with the sha256 |
| `load_flex(path, verify)` | Processed CSV | The table, after checking it against the hash in the manifest |

Why: cleaning makes no modeling decision. It keeps all 274 rows and records problems as flag columns
(`duplicate_of`, `high_pressure`, `fiber` unknown, ...), so the filters of each model are explicit and
counted (D-03). `load_flex` refuses a table that does not match its manifest, which ties every model to one
data hash. This module was written before this task; only its comments were removed (T-01).

### 4.2 `flex_check.py`: data check report

`run_check()` writes `reports/flex_data_check.md`: outcome counts, what "Disruption" means, extinction
diameters, missing values, coverage of the Moon and ISS scenarios, variables that are not in the table and
candidate training sets. It is the evidence base of D-01 to D-05. `in_moon_box` is reused by the model card.
Only its comments were removed (T-01).

### 4.3 `config.py`: experiment configuration

| Function | Input | Output |
|---|---|---|
| `load_config(path)` | TOML path, relative to `train/` | `ExperimentConfig` (path, sha256 of the file, values, reference config) |
| `validate_config(values)` | Parsed values | Raises `ValueError` for a missing section, an unknown role or task, no single baseline, a calibrated model without `[calibration]`, a classification without exactly two classes, a regression without `target.output` and `cv.stratify_bins` |
| `resolve_path(relative)` | Path | Absolute path under `train/` |

`ExperimentConfig` exposes `name`, `version`, `model_version`, `role`, `task`, `decisions`,
`baseline_name`, `candidate_names` and `feature_columns`.

Why: a config with `[comparison].reference_config` inherits the reference sections and overrides keys one
level deep (T-02), so the sensitivity target cannot drift from the reference models and filters. The sha256
of the file is written to every report and to `model_info.json`.

Tests: `tests/training/test_config.py`.

### 4.4 `datasets.py`: rows, target and groups

| Function | Input | Output |
|---|---|---|
| `load_training_data(config)` | Config | `TrainingData`, after `load_flex` verified the hash |
| `build_training_data(table, config, source_sha256)` | Any table with the FLEX schema | `TrainingData` |
| `apply_exclusions(table, exclusions)` | Table and ordered filters | Kept rows and the number of rows each filter removed |
| `exclusion_mask(column, rule, value)` | Column and rule (`not_null`, `is_null`, `is_true`, `equals`, `not_equals`) | Boolean mask of rows to remove |
| `load_source_table(config)` | Config | The full processed table and its sha256 |

`TrainingData` holds `features` (the pipeline inputs), `target`, `groups` (`atmosphere_id`), `rows` (the
kept rows with every column), `exclusions`, `source_rows`, `source_sha256`, and the properties `ids` and
`row_keys`. All frames share the same 0-based index.

Why: filters are data, not code, and are counted in order so the counts add up (D-03, T-03). A class that
has no entry in the class map stops the run, so a new test-end label cannot slip in silently.

Tests: `tests/training/test_datasets.py`; on the real table, `test_training_rows_match_decision_d03`,
`test_sensitivity_rows_match_decision_d02` and `test_extinction_rows_match_decision_d08` in
`tests/training/test_flex_integration.py`.

### 4.5 `splits.py`: grouped splits

| Function | Input | Output |
|---|---|---|
| `repeated_group_splits(labels, groups, cv)` | Strata, groups, `[cv]` | List of `Split` (repeat, fold, seed, train rows, test rows) |
| `stratification_labels(target, config)` | Target | The classes, or quantile bins of the target for a regression |
| `grouped_calibration_splits(target, groups, calibration, seed)` | Training rows of one outer fold | Index pairs (fit rows, calibration rows) that share no group |

Why: tests of one chamber fill are near duplicates, so they must stay on the same side of every split
(D-05). Repeat `r` uses `seed + r` (T-05). A regression is stratified on target bins so it can use the same
splitter (T-16). A calibration split that lacks a class on either side raises an error instead of producing
a degenerate calibrator.

Tests: `tests/training/test_splits.py`.

### 4.6 `features.py`: preprocessing

`build_preprocessor(features, spline)` returns a `ColumnTransformer` with: one-hot fuel; standardized
numeric inputs and fiber flag; `po2` as the row product of `o2` and `pressure_atm`, then standardized; and,
for the spline model, B-splines that replace the linear terms of the listed columns.

Why: `row_product()` is `FunctionTransformer(numpy.prod, ...)`. A numpy function can be saved and loaded by
name, so the pipeline needs neither a lambda nor a class from this package (D-09, T-07).

### 4.7 `models.py`: model registry

| Function | Input | Output |
|---|---|---|
| `build_model(model, features, calibration, calibration_splits)` | One `[models.<name>]` table | Unfitted pipeline: preprocessing + estimator, wrapped in `CalibratedClassifierCV` when `calibrated = true` |
| `fit_model(model_name, config, features, target, groups, seed)` | Training rows | Fitted pipeline, with grouped calibration splits computed on those rows |
| `predict_values(pipeline, features, config)` | Fitted pipeline | Probability of the positive class, or the clipped regression value |
| `ESTIMATORS` | — | Names allowed in the config: logistic regression, ridge, and the two gradient-boosting estimators |

Why: every model has the common scikit-learn interface (`fit`, `predict`, `predict_proba`), so the
evaluation, the export and the sensitivity check work with any of them. Calibration receives precomputed
grouped splits (T-06). Adding an algorithm means one entry in `ESTIMATORS` and one table in the config.

### 4.8 `evaluate.py`: cross-validation, metrics and selection

| Function | Input | Output |
|---|---|---|
| `evaluate_models(data, config)` | Training data and config | `Evaluation`: `fold_metrics`, `repeat_metrics`, `predictions`, `selection`, `metric_names` |
| `evaluation_splits(data, config)` | Same | The outer splits, identical for every model |
| `classification_metrics(truth, positive_probability, target)` | Labels and probabilities | macro-F1, recall per class, Brier score, ROC-AUC |
| `regression_metrics(truth, predicted)` | Values | MAE, RMSE, R2 |
| `pooled_repeat_metrics(predictions, target, config)` | Out-of-fold predictions | One metric row per model and repeat |
| `paired_comparison(...)`, `select_model(repeat_metrics, config)` | Metrics per repeat | The comparison with the baseline and the selected model with its reason |
| `absolute_error_by_region(evaluation, data, config)` | Regression evaluation | Mean absolute error per pressure level and suppressant |

Why: the loop over folds is written by hand because each outer fold needs its own grouped calibration
splits (T-06). Every model is fitted and tested on exactly the same rows. The selection rule is applied
mechanically to the pooled out-of-fold predictions of each repeat (T-04); a tie keeps the baseline. A metric
that is undefined on a fold (a fold with one class) is stored as missing, never as 0.

Tests: `tests/training/test_selection.py` (rule and metrics, with hand-computed cases),
`tests/training/test_export.py::test_two_evaluations_give_identical_metrics_and_reports`,
`tests/training/test_regression.py`.

### 4.9 `support.py`: how much real data supports a prediction

| Function | Input | Output |
|---|---|---|
| `fit_support(features, support)` | Training features and `[support]` | Dictionary with one partition per fuel: scaler, neighbor index, threshold, reference rows |
| `query_support(bundle, feature_row)` | One feature row | `within_joint_range`, `support_n`, the nearest reference rows and their distances |
| `confidence_level(in_tested_range, support_n, top_probability, rule)` | Support result | `low`, `medium` or `high` |
| `support_thresholds(bundle)` | Bundle | Threshold per fuel |

Why: every input can be inside its own range while the combination was never tested, so the range check
is a distance to the nearest real test (D-07). The threshold is the 95th percentile of the distance from
each training test to its nearest other test: a query is "in range" when it is as close to real data as the
real tests are to each other. The bundle is a plain dictionary of scikit-learn objects (T-10, D-09).

Tests: `tests/training/test_support.py` recomputes the threshold and `support_n` by brute force and checks
every boundary of the confidence rule.

### 4.10 `inference.py`: reference prediction

| Function | Input | Output |
|---|---|---|
| `load_bundle(artifact_dir)` | Artifact folder | Dictionary with `model`, `support`, `reference` and `info` |
| `reference_predict(bundle, inputs)` | Exposed inputs | The response |
| `validate_inputs(info, inputs)` | Exposed inputs | The inputs outside their tested range; raises for invalid requests |
| `build_feature_row(info, inputs)` | Exposed inputs | One-row frame with the pipeline columns |
| `check_constraints(info, feature_row)` | Feature row | Raises when `o2 + co2 + he > 1` |

Exposed inputs: `fuel`, `pressure_atm`, `o2`, `suppressant`, `suppressant_fraction`, `droplet_d0_mm`.

Response of `flex-outcome`: `outcome`, `probabilities`, `in_tested_range`, `out_of_range_fields`,
`support_n`, `nearest_tests`, `confidence`, `model_version`, `training_source`. `flex-extinction` returns
`extinction_diameter_ratio` and `extinction_diameter_mm` in place of the first two.

Why: this function uses only the four artifact files, never the TOML, so it is exactly what `ml/` has to
reproduce. It translates the suppressant choice into the `co2` and `he` columns, fixes `fiber` to false
(D-04), flags numbers outside their tested range and rejects structurally invalid requests (T-12).

Tests: `tests/training/test_inference.py`; with the real data, the three smoke tests in
`tests/training/test_flex_integration.py`.

### 4.11 `reporting.py`: generated reports

`evaluation_report`, `model_card` and `sensitivity_report` build Markdown from the objects above;
`write_evaluation_report` writes the first one. `flex_coverage_facts` computes the coverage table of the
model card (pressure levels, highest O2 with a suppressant, Moon-like tests, fiber rows, cool-flame
mentions).

Why: every number in a report is computed here, and the reports carry no timestamp, so two runs give
identical files (T-15). The texts on intended use and limitations come from the config.

### 4.12 `export.py`: artifact and model card

| Function | Input | Output |
|---|---|---|
| `export_experiment(config, artifacts_root, reports_root)` | Publishable config | Runs the whole export on the processed table |
| `export_from_data(config, data, source_table, ...)` | Training data | Evaluates, selects, refits on all rows, writes the artifact, the evaluation report and the model card |
| `build_model_info(config, data, evaluation, support_bundle)` | Results | The content of `model_info.json` |
| `describe_inputs(config, data)` | Training rows | Exposed inputs with ranges computed from the data, rounded outward (T-11) |
| `round_outward(value, decimals, upward)` | Number | The number rounded away from the tested interval |
| `run_reference_queries(bundle, config)` | Loaded artifact | Answers to the queries of `[[report.reference_queries]]` |

`model_info.json` contains: `model_version`, `task`, `algorithm`, class labels or output names, `inputs`,
`input_mapping`, `constraints`, `fixed_inputs`, `feature_columns`, `feature_dtypes`, `outputs`,
`training_source`, `support`, `cv`, `decisions`, `config_sha256`, `versions`, `code_version`, `files`,
`created`.

Why: the export always evaluates first, so a model cannot be published without its comparison with the
baseline. It refuses a config whose role is `sensitivity` (D-02). The reference queries are answered from
the files just written, which also proves that the artifact loads.

Tests: `tests/training/test_export.py`, including the load in an isolated interpreter (D-09).

### 4.13 `sensitivity.py`: target sensitivity check

`compare_targets(config)` loads the reference and the sensitivity data, evaluates the reference target,
takes the selected algorithm, trains it under the sensitivity target on the splits derived from the
reference splits, and writes `reports/<reference>_target_sensitivity.md` with a mechanical verdict (D-02,
T-14). `reference_positions` maps each sensitivity row to its reference row and rejects rows that are not in
the reference set. `spearman_correlation` is the Pearson correlation of average ranks.

Tests: `tests/training/test_sensitivity.py`.

### 4.14 `cli.py` and `main.py`: commands

`main.py` is the shared entry point and was not changed. `cli.py` registers the `train` commands and imports
the heavy modules only inside each command.

| Command | Effect |
|---|---|
| `train fetch-flex` | Download the table and the TestSheets |
| `train prepare-flex` | Clean and write the processed table with its hash |
| `train check-flex` | Write the data check report |
| `train evaluate --config <toml>` | Cross-validate, apply the selection rule, write the evaluation report |
| `train export --config <toml>` | Evaluate, refit, write the artifact, the evaluation report and the model card |
| `train compare-targets [--config <toml>]` | Run the target sensitivity check |

## 5. Tests

`tests/training/` holds the tests. Unit tests use a synthetic table built in `conftest.py` and shortened
configs (fewer repeats and boosting iterations); they need neither the network nor the data. The tests in
`test_flex_integration.py` use the real table and are skipped when it is missing (T-17).

| File | What it protects |
|---|---|
| `test_config.py` | Config loading, inheritance of the sensitivity config, validation errors |
| `test_datasets.py` | Ordered exclusion counts, class map, feature types, alignment, subset for option B |
| `test_splits.py` | No group shared between folds, nor between fit and calibration; reproducible splits |
| `test_selection.py` | Selection rule edge cases (ties, one metric only, two winners), metrics by hand |
| `test_support.py` | Threshold, `support_n`, nearest tests, joint range, confidence boundaries |
| `test_inference.py` | Response fields and types, suppressant mapping, fiber never exposed, invalid requests |
| `test_export.py` | Artifact files, `model_info.json`, outward rounding, isolated load, determinism |
| `test_sensitivity.py` | Row mapping, Spearman, mechanical verdict |
| `test_regression.py` | Regression path: rows, strata, metrics, clipping, response, `model_info.json` |
| `test_flex_integration.py` | Row counts of D-03, D-02 and D-08, real group separation, the three smoke tests |

## 6. Decision to code matrix

| Decision | Where it is implemented |
|---|---|
| D-01 binary target | `flex_outcome.toml` `[target.class_map]`; `datasets.build_target`; `evaluate.class_labels` |
| D-02 sensitivity check | `flex_outcome_sensitivity.toml`; `sensitivity.compare_targets`; `export.export_from_data` (role check) |
| D-03 exclusions and data-driven ranges | `[[dataset.exclusions]]`; `datasets.apply_exclusions`; `export.describe_inputs` |
| D-04 inputs and interface | `[features]`, `[interface]`; `features.build_preprocessor`; `inference.build_feature_row`, `validate_inputs`, `check_constraints` |
| D-05 grouped repeated CV | `[cv]`, `[calibration]`; `splits.repeated_group_splits`, `grouped_calibration_splits`; `evaluate.evaluate_models` |
| D-06 models and selection | `[models.*]`, `[selection]`; `models.build_model`; `evaluate.select_model` |
| D-07 support and confidence | `[support]`; `support.fit_support`, `query_support`, `confidence_level`; `inference.reference_predict` |
| D-08 extinction regression | `flex_extinction.toml`; `evaluate.regression_metrics`; `models.predict_values`; `inference.predict_outputs` |
| D-09 portable artifacts | `features.row_product`; `support.fit_support`; `export.write_artifact` |
| T-02 config inheritance | `config.load_config`, `_merge_sections` |
| T-03 ordered filter counts | `datasets.apply_exclusions` |
| T-04 comparison unit | `evaluate.pooled_repeat_metrics`, `paired_comparison` |
| T-05 seeds | `splits.repeated_group_splits`; `models.fit_model` |
| T-06 grouped calibration | `splits.grouped_calibration_splits`; `models.build_model` |
| T-07 preprocessing | `features.build_preprocessor` |
| T-09 decision threshold | `evaluate.predicted_labels`; `inference.predict_outputs` |
| T-10 support details | `support.fit_partition`; `inference.reference_predict` |
| T-11 outward rounding | `export.round_outward`, `tested_range` |
| T-12 errors vs. flags | `inference.validate_inputs`, `check_constraints` |
| T-13 `training_source` shape | `inference.reference_predict`; `export.describe_training_source` |
| T-14 target comparison | `sensitivity.compare_targets_from_data`, `spearman_correlation` |
| T-15 deterministic reports | `reporting.py`; `export.run_reference_queries` |
| T-16 regression details | `splits.stratification_labels`; `models.predict_values`; `support.confidence_level` |
| T-17 test setup | `tests/training/conftest.py` |

## 7. Reproduce everything from scratch

Run from `train/`, in an environment with the packages of `environment.yml`:

```
python main.py train fetch-flex
python main.py train prepare-flex
python main.py train check-flex
python main.py train export --config configs/training/flex_outcome.toml
python main.py train compare-targets
python main.py train export --config configs/training/flex_extinction.toml
python -m pytest tests/training
```

`train evaluate --config <toml>` repeats only the cross-validation and its report. The export of
`flex-outcome` takes about one minute; the other commands are faster.

Load a model without this package:

```
import json, joblib, pandas as pd
info = json.load(open("artifacts/models/flex-outcome-0.1/model_info.json"))
model = joblib.load("artifacts/models/flex-outcome-0.1/model.joblib")
row = pd.DataFrame([{"fuel": "methanol", "pressure_atm": 1.0, "o2": 0.21, "co2": 0.0, "he": 0.0,
                     "d0_mm": 3.0, "fiber": False}])[info["feature_columns"]]
print(dict(zip(model.classes_, model.predict_proba(row)[0])))
```

## 8. Verification

Every check below was run on 2026-11-14 on the final code, with Python 3.13.9, scikit-learn 1.7.2, numpy 2.3.5
and pandas 2.3.3. The results are copied from the output of the commands named in each row.

| # | Check | How it was run | Result |
|---|---|---|---|
| 1 | Cleaning unchanged after removing comments | `train prepare-flex`, then sha256 of `flex_tests.csv` against the value saved before any edit; `train check-flex`, then byte comparison with the report saved before any edit; syntax trees of `flex_data.py` and `flex_check.py` compared before and after, ignoring docstrings | Pass. Same sha256 (`abb213da368e…8f74`), identical report, identical syntax trees |
| 2 | No comments, no docstrings | A script using `ast` and `tokenize`, run with `python3 -I` on every `.py` file in `src/flame/training/` and `tests/training/`; `grep "#"` on the three TOML files and `environment.yml` | Pass. 26 Python files clean and at most 120 columns wide; no `#` in the configs |
| 3 | Tests | `python3 -m pytest tests/training` | Pass. 99 passed, none skipped, no network used |
| 4 | Deterministic evaluation | `train export` followed by `train evaluate` for each publishable config, then byte comparison of the two evaluation reports | Pass for `flex-outcome` and for `flex-extinction`: identical reports |
| 5 | Artifact loads without this package | `python3 -I` started in another directory loads `model.joblib` and `support.joblib` of both models, predicts one row and queries the neighbor index | Pass for both models. No `flame` module was imported |
| 6 | Smoke queries on the real `flex-outcome` artifact | `inference.reference_predict` on three inputs; the same three are reference queries in the model card and tests in `test_flex_integration.py` | Pass. (a) methanol, 0.7 atm, O2 0.34, CO2 0.20: out of range, `support_n` 0, confidence low. (b) methanol, 1.0 atm, O2 0.21, no suppressant, D0 3.0: in range, `support_n` 10, confidence medium. (c) methanol, 0.69 atm, O2 0.34, no suppressant: in range, `support_n` 6, confidence high, prediction `no_extinction` |
| 7 | Row counts of D-03 | `datasets.load_training_data` on the three configs | Pass. Reference 251 rows (extinction 171, no_extinction 80); sensitivity 198 (171, 27); extinction regression 156 |
| 8 | Git state | `git rev-parse HEAD` against the value saved at the start; `git status --short`; byte comparison of `main.py` with the copy saved at the start | Pass. HEAD is still `32be15c`; nothing was staged or committed; `main.py` was not touched in this task |

Notes on check 8: `git status` lists `main.py` as modified because it already was before this task started
(the `train` command group was registered earlier). `data/` and `artifacts/` are ignored by git, so the
processed table and the two model folders do not appear in `git status`.

Notes on check 6: case (c) is in range with high confidence. `CLAUDE.md` §11.8 expected a low-confidence
answer at the Moon corner; that expectation predates the data check (O-04). The case without supporting data
is (a), the Moon-like atmosphere with a suppressant.
