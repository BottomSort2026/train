# Decision log: FLEX training pipeline

This log records every decision behind the FLEX (PSI-69) training code in `train/`: what was decided, what
else was considered, why, the evidence, the files it touches and how to undo it.

- **ADOPTED** decisions (`D-xx`) were taken by the team before the code was written. The code applies them as
  given.
- **TECHNICAL** decisions (`T-xx`) were taken while writing the code, inside the scope of the adopted ones.
- **Open items** (`O-xx`) are not decided here. They need the team.

Numbers quoted in this file are copied from generated reports, and each one names its source. The reports
are regenerated with the commands in `reports/code_walkthrough.md`.

Source reports:

| Short name | File |
|---|---|
| data check | `reports/flex_data_check.md` |
| outcome evaluation | `reports/flex-outcome_evaluation.md` |
| outcome model card | `reports/flex-outcome-0.1_model_card.md` |
| sensitivity report | `reports/flex-outcome_target_sensitivity.md` |
| extinction evaluation | `reports/flex-extinction_evaluation.md` |
| extinction model card | `reports/flex-extinction-0.2_model_card.md` |

## Adopted decisions

### D-01 Binary target for `flex-outcome` v0.1

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | `flex-outcome` predicts two classes: `extinction` and `no_extinction`. Tests that ended in `disruption` or `completion` count as `no_extinction`. The positive class is `extinction`. |
| Options considered | (A) the three published classes; (B) drop the disruption tests and predict extinction vs. completion; (C) count disruption as "the atmosphere did not put the flame out". C was chosen. |
| Reason | The fire-safety question is whether the atmosphere puts the flame out. A disruption is a mechanical event: the droplet breaks apart or leaves its support fiber, and the flame ends because of that. Dropping those tests (B) would leave almost no data for the Moon-like atmosphere. |
| Evidence | 52 of the 61 disruption comments mention the disruption; the median droplet size at the end is 0.29 of the initial diameter for disruptions and 0.72 for extinctions; 18 disruptions happened at O2 >= 0.30, against 2 extinctions (data check §2). Option B keeps 2 Moon-like rows, option C keeps 16 (data check §7). Classes after the exclusions: extinction 171, no_extinction 80 (outcome evaluation, run table). |
| Known risk | A disruption early in the burn can hide an extinction that would have happened later. D-02 measures how much this assumption changes the result. |
| Affected files | `configs/training/flex_outcome.toml` (`[target]`, `[target.class_map]`), `src/flame/training/datasets.py` (`build_target`), `src/flame/training/evaluate.py` (`class_labels`, `classification_metrics`). |
| How to revert | Edit `[target.class_map]`. A three-class target also needs a new metric and selection rule, because `validate_config` accepts exactly two classes. |

### D-02 Option B is a sensitivity check only

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | Option B (extinction vs. completion, without the disruption tests) is evaluated and compared with the D-01 target, and is never exported. Its splits are derived from the reference splits: the same atmosphere groups sit in the same folds. The two targets are called "consistent" when the absolute ROC-AUC difference on the shared test rows is at most 0.05 and the Spearman correlation between the probabilities of the two final models on the shared rows is at least 0.80. The report states the verdict mechanically. |
| Options considered | Not checking the D-01 assumption; checking it by eye. A criterion fixed before the run was chosen so the verdict cannot be adjusted afterwards. |
| Reason | D-01 rests on one assumption (a disruption means the flame was not put out). Training without the disruption tests shows whether the conclusions depend on it. |
| Evidence | Reference rows 251, sensitivity rows 198 (extinction 171, no_extinction 27), shared rows 198. ROC-AUC difference 0.0164, Spearman 0.9822: **consistent** (sensitivity report). |
| Affected files | `configs/training/flex_outcome_sensitivity.toml`, `src/flame/training/sensitivity.py`, `src/flame/training/reporting.py` (`sensitivity_report`), `src/flame/training/cli.py` (`compare-targets`), `src/flame/training/export.py` (refuses the role `sensitivity`). |
| How to revert | Delete the sensitivity config and the `compare-targets` command. To change the criterion, edit `[comparison]` before running, and record the change here. |

### D-03 Rows excluded from training

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | A row is excluded when it is a duplicate (`duplicate_of` not null), has no initial diameter (`d0_mm` null), is a high-pressure test (`high_pressure`), or has an unknown fiber flag (`fiber` null). Calibration tests stay. Every range (pressure, O2, fractions, D0) is computed from the remaining rows and never written by hand. |
| Options considered | Keeping the high-pressure tests; imputing the missing diameters; treating an unknown fiber as a free droplet. |
| Reason | The initial diameter is a model input, so a row without it cannot be used. Above 1.5 atm every test used the same mixture, so the effect of pressure cannot be separated from the effect of that mixture. A duplicated test would count twice. The fiber flag is a model input (D-04). |
| Evidence | Rows removed, in order: duplicate_test 1, missing_initial_diameter 12, high_pressure 9, unknown_fiber 1; 251 of 274 rows remain (outcome evaluation, "Rows excluded"). The 9 tests above 1.5 atm all have O2 0.21 with 70 % CO2 (data check §5). The duplicate is test `193F001`, table rows 70 and 73, and three tests have no TestSheet and therefore no fiber flag (data check, cleaning log). Expected counts were about 251 (171 / 80) and 198 (171 / 27) for option B; the code gives exactly those (outcome evaluation; sensitivity report; `tests/training/test_flex_integration.py`). |
| Affected files | `configs/training/*.toml` (`[[dataset.exclusions]]`), `src/flame/training/datasets.py` (`apply_exclusions`, `exclusion_mask`), `src/flame/training/export.py` (`describe_inputs`, `tested_range`). |
| How to revert | Remove or add `[[dataset.exclusions]]` entries. Each filter reports how many rows it removes. |

### D-04 Model inputs and exposed interface

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | The pipeline reads `fuel`, `pressure_atm`, `o2`, `co2`, `he`, `d0_mm` and `fiber`, and derives `po2 = o2 * pressure_atm`. N2 is the remainder of the mixture. The exposed interface is `fuel`, `pressure_atm`, `o2`, `suppressant` (none / co2 / he), `suppressant_fraction` (with a tested range per suppressant) and `droplet_d0_mm`, with the constraint `o2 + co2 + he <= 1`. `fiber` is fixed to false and not exposed: every prediction is for a free droplet. The imposed slow flow is not used, and the extinction diameters that appear only in the TestSheets are not filled in. |
| Options considered | Two independent sliders for CO2 and He; dropping the fiber-supported tests; adding the imposed flow from the comments; filling missing diameters from the TestSheets. |
| Reason | CO2 and He were never mixed in one test, so two free sliders would let a user ask for mixtures nobody tested. The fiber changes how a droplet burns, so the model must know about it, but a real fire has no fiber. The flow is only mentioned in free-text comments. The published table is the official source. |
| Evidence | No FLEX test has both CO2 and He (`tests/training/test_flex_integration.py::test_co2_and_helium_are_never_mixed`). 54 of the 251 training rows are fiber-supported (outcome model card, coverage facts). 44 comments mention translation (data check §6). 24 extinction diameters exist only in the TestSheets (data check, cleaning log). Tested ranges per input, including the fraction range per suppressant, are in the outcome model card ("Inputs and tested ranges"). |
| Affected files | `configs/training/*.toml` (`[features]`, `[interface]`), `src/flame/training/features.py`, `src/flame/training/export.py` (`describe_inputs`, `describe_input_mapping`), `src/flame/training/inference.py` (`validate_inputs`, `build_feature_row`, `check_constraints`). |
| How to revert | To expose `fiber`, remove it from `[interface.fixed_inputs]` and add it to `[[interface.direct_inputs]]`. To drop the derived feature, remove `[[features.products]]`. |

### D-05 Grouped, repeated cross-validation

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | `StratifiedGroupKFold` with groups = `atmosphere_id`, 5 folds x 10 repeats, shuffled, with the seed in the config. Calibration also respects the groups: one `atmosphere_id` is never used both to fit and to calibrate a model. |
| Options considered | Random splits; groups by fuel x diluent x pressure level. |
| Reason | Tests from one chamber fill share the same atmosphere and are near duplicates, so a random split lets the model answer almost from memory. The simulator predicts atmospheres that were never tested, which is what a split by atmosphere measures. Groups by fuel x diluent x pressure would be too few and would measure extrapolation to whole families. |
| Evidence | 44 atmosphere groups in the training rows (outcome evaluation, run table); group sizes 1 to 27, median 5, and 24 groups contain more than one outcome (data check §7). Group separation is tested in `tests/training/test_splits.py` and, on the real rows, in `tests/training/test_flex_integration.py`. |
| Affected files | `configs/training/*.toml` (`[cv]`, `[calibration]`), `src/flame/training/splits.py`, `src/flame/training/evaluate.py` (`evaluation_splits`, `evaluate_models`), `src/flame/training/models.py` (`fit_model`). |
| How to revert | Change `[cv]` in the config. Another grouping only needs a different `dataset.group_column`. |

### D-06 Models and selection rule for `flex-outcome`

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | Baseline: logistic regression (one-hot fuel, standardized numbers, balanced class weights). Candidates: `logreg_spline` (splines on `o2`, `d0_mm`, `co2`, `he`; pressure stays linear) and `hgb` (shallow `HistGradientBoostingClassifier`, balanced class weights, sigmoid calibration with grouped splits). Hyperparameters are fixed in the TOML and not searched. A candidate is published only if it beats the baseline on macro-F1 **and** on Brier score (paired means on the same splits). If both candidates qualify, the one with the lower Brier score wins. If none does, the baseline is published. No monotonic constraints. |
| Options considered | Deep learning (too few tests); hyperparameter search (it would tune to the validation folds); monotonic constraints (directions not yet confirmed in the FLEX papers, see O-03). |
| Reason | With about 250 rows a simple, regularized model generalizes better and can be explained. Fixing the rule and the hyperparameters before looking at results keeps the comparison honest. |
| Evidence | Pooled out-of-fold results, mean over 10 repeats: `logreg` macro-F1 0.7933, Brier 0.1327; `logreg_spline` 0.7907, 0.1348; `hgb` 0.7566, 0.1396. No candidate beats the baseline on either metric, so **`logreg` is published** (outcome evaluation, "Paired comparison with the baseline" and "Selection"). |
| Affected files | `configs/training/flex_outcome.toml` (`[models.*]`, `[calibration]`, `[selection]`), `src/flame/training/models.py`, `src/flame/training/features.py`, `src/flame/training/evaluate.py` (`select_model`, `paired_comparison`). |
| How to revert | Edit `[models.*]` or `[selection]`. Any change made after seeing results must be recorded here as a new decision. |

### D-07 Support, range flag and confidence

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | One nearest-neighbor index per fuel over `pressure_atm`, `o2`, `co2`, `he`, `d0_mm` (without `fiber`), standardized. The joint range threshold is the 95th percentile of the distance from each training row to its nearest other row, per fuel. `support_n` is the number of training rows of the same fuel within the threshold. `nearest_tests` lists the 5 closest tests (`cir_id`, `testsheet_id`, `test_end`, distance). `out_of_range_fields` lists the inputs outside their own minimum and maximum. `confidence` is `low` if out of range or `support_n` < 3; `high` if in range, `support_n` >= 5 and the probability of the predicted class is at least 0.7; otherwise `medium`. All these numbers live in the config. |
| Options considered | Per-input range checks only; one index for both fuels with a one-hot fuel column. |
| Reason | Every input can be inside its own range while the combination was never tested. Separate indexes per fuel avoid inventing a distance between two fuels. |
| Evidence | Thresholds: heptane 0.7236, methanol 0.8300 (outcome model card, "Support and confidence"). Reference queries (outcome model card): Moon-like atmosphere with CO2 is out of range with `support_n` 0 and confidence low; ISS-like air is in range with `support_n` 10; Moon-like atmosphere without suppressant is in range with `support_n` 6; a pressure halfway between the two tested levels is out of range. |
| Known consequence | The joint check rejects a pressure only when it is far from both tested levels. A pressure close to one of the levels passes although only two levels were tested (see O-08). |
| Affected files | `configs/training/*.toml` (`[support]`), `src/flame/training/support.py`, `src/flame/training/inference.py` (`reference_predict`), `src/flame/training/export.py` (`describe_support`, `support_reference`). |
| How to revert | Edit `[support]`. The rule itself is `confidence_level` in `support.py`. |

### D-08 `flex-extinction` v0.2

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | Regression of `dext_over_d0` (extinction diameter over initial diameter) on the rows whose test end is `extinction` and that have a numeric `dext_mm` and `d0_mm`, after the D-03 exclusions. Baseline: Ridge. Candidate: `HistGradientBoostingRegressor`. Same cross-validation as D-05. The candidate is published only if its paired mean MAE is lower than the baseline's. The model is conditional on the flame going out, and its model card carries the heptane cool-flame caveat. It was built only after v0.1 passed every verification. |
| Options considered | Treating a completion as a ratio of 0 in the same model; predicting the diameter in millimetres. |
| Reason | The ratio measures how early the flame goes out: near 1 means it barely outlived ignition. Dividing by the initial diameter makes droplets of different sizes comparable. Completions and extinctions are different regimes, and `flex-outcome` already separates them. |
| Evidence | 156 training rows; `hgb` MAE 0.0998 against `ridge` 0.1080, better in 9 of 10 repeats, so `hgb` is published (extinction evaluation). Details under "Results of D-08" below. |
| Affected files | `configs/training/flex_extinction.toml`, plus the task-generic code: `src/flame/training/evaluate.py` (`regression_metrics`, `absolute_error_by_region`), `src/flame/training/splits.py` (`stratification_labels`), `src/flame/training/models.py` (`predict_values`), `src/flame/training/inference.py` (`predict_outputs`). |
| How to revert | Delete `configs/training/flex_extinction.toml` and its artifact. No other model depends on it. |

### D-09 Artifacts and portability

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | ADOPTED |
| Decision | Artifacts go to `artifacts/models/<model>-<version>/` (the layout of this repository, not the one in `CLAUDE.md`). The pipeline uses only scikit-learn and numpy components: no custom classes, no lambdas, no functions from `flame`. It must load with `python3 -I` without `src` on the path. |
| Options considered | Custom transformer classes shipped as an installable package shared with `ml/`. |
| Reason | `ml/` then needs only the same scikit-learn version to load the model, and never reimplements the feature code. |
| Evidence | `tests/training/test_export.py::test_artifact_loads_in_a_clean_interpreter_without_the_training_code` loads the model and the support index in an isolated interpreter and checks that no `flame` module is imported. The same check was run by hand on the real artifact (see `reports/code_walkthrough.md`, "Verification"). |
| Affected files | `src/flame/training/features.py` (`row_product` uses `numpy.prod`), `src/flame/training/models.py`, `src/flame/training/support.py` (plain dictionary of scikit-learn objects), `src/flame/training/export.py` (`write_artifact`). |
| How to revert | Not recommended. A custom class in the pipeline would force `ml/` to install this package. |

## Technical decisions

### T-01 Comments and docstrings removed from the existing files

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | `flex_data.py`, `flex_check.py` and `cli.py` lost their comments and docstrings, and one line longer than 120 columns in `flex_check.py` was wrapped. Nothing else changed. |
| Options considered | Leaving the existing files as they were. |
| Reason | The rule "no comments in any file that is written or touched" covers these files. The explanation of the code now lives in `reports/code_walkthrough.md`. |
| Evidence | The syntax trees before and after are identical apart from docstrings; `prepare-flex` produces the same sha256 for `flex_tests.csv`; `check-flex` produces an identical report (`reports/code_walkthrough.md`, "Verification", checks 1 and 2). |
| Affected files | `src/flame/training/flex_data.py`, `src/flame/training/flex_check.py`, `src/flame/training/cli.py`. |
| How to revert | Not applicable: behavior is unchanged. |

### T-02 One TOML per experiment; the sensitivity config only states its differences

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Each experiment is one TOML file read with `tomllib`. A config with `[comparison].reference_config` inherits every section of the reference config, and its own sections override keys one level deep. The sensitivity config therefore contains only its role, its extra exclusion and its class map. |
| Options considered | Copying the whole reference config into the sensitivity config. |
| Reason | Two copies of the models and of the D-03 filters would drift apart, and the comparison is only valid if both targets use exactly the same models and splits. TOML needs no new dependency. |
| Evidence | `tests/training/test_config.py::test_sensitivity_config_inherits_models_and_overrides_the_target`. |
| Affected files | `src/flame/training/config.py` (`load_config`, `_merge_sections`), `configs/training/flex_outcome_sensitivity.toml`. |
| How to revert | Write the sensitivity config in full and drop `_merge_sections`. |

### T-03 Exclusion filters are applied in order and counted once

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Filters run in the order of the config. A row that matches two filters is counted under the first one. |
| Options considered | Counting every filter on the full table, with overlaps. |
| Reason | The counts then add up to the number of rows removed, which is what the model card and `model_info.json` must show. |
| Evidence | `tests/training/test_datasets.py::test_a_row_matching_two_filters_is_counted_once`. |
| Affected files | `src/flame/training/datasets.py` (`apply_exclusions`). |
| How to revert | Change `apply_exclusions`. |

### T-04 Models are compared on the pooled out-of-fold predictions of each repeat

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | The paired comparison of D-06 and D-08 uses one value per repeat: the metric computed on the out-of-fold predictions of all rows of that repeat. The per-fold mean and standard deviation are also reported. "Beats" means a strictly better mean of the paired differences; a tie keeps the baseline. |
| Options considered | Pairing on the 50 individual folds. |
| Reason | Folds are groups of whole atmospheres, so their sizes and class mixes differ a lot, and macro-F1 on a small fold with very few `no_extinction` rows is unstable. A repeat predicts every row exactly once, so its metric is always defined. |
| Evidence | Both views give the same selection: per fold, `logreg` has macro-F1 0.7668 and Brier 0.1259, better than both candidates, as in the pooled view (outcome evaluation, both metric tables). Rule edge cases: `tests/training/test_selection.py`. |
| Affected files | `src/flame/training/evaluate.py` (`pooled_repeat_metrics`, `paired_comparison`, `select_model`). |
| How to revert | Pivot `fold_metrics` by `(repeat, fold)` in `paired_comparison`. |

### T-05 Seeds

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Repeat `r` uses `cv.seed + r` for its outer split and for the calibration splits inside its folds. The final fit on all rows uses `cv.seed`. Gradient boosting has `random_state` in its parameters. |
| Options considered | A single seed for every repeat (the repeats would be identical). |
| Reason | Repeats must differ from each other and still be reproducible. |
| Evidence | Two consecutive evaluations give identical reports (`reports/code_walkthrough.md`, "Verification", check 4; `tests/training/test_export.py::test_two_evaluations_give_identical_metrics_and_reports`). |
| Affected files | `src/flame/training/splits.py`, `src/flame/training/models.py` (`fit_model`). |
| How to revert | Change `repeated_group_splits`. |

### T-06 Grouped calibration through precomputed splits

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | For a calibrated model, the grouped calibration splits are computed on the training rows of each outer fold and passed to `CalibratedClassifierCV(cv=...)` as a list of index pairs. This is why the evaluation is a manual loop over folds. The preprocessing step (scaling, one-hot encoding, spline knots) is fitted on the whole outer training fold. |
| Options considered | scikit-learn metadata routing of `groups` to the internal splitter; a custom wrapper class (forbidden by D-09); the default non-grouped internal split (it would break D-05). |
| Reason | A list of splits needs no experimental API and no custom class, and it stays inside the saved pipeline as plain arrays. The preprocessing statistics do not use the target, so fitting them on the outer training fold leaks no label information. |
| Evidence | `tests/training/test_splits.py::test_fitted_calibrated_model_keeps_groups_apart_inside_each_outer_fold` reads the splits back from a fitted model. |
| Affected files | `src/flame/training/splits.py` (`grouped_calibration_splits`), `src/flame/training/models.py` (`build_model`, `fit_model`), `src/flame/training/evaluate.py` (`evaluate_models`). |
| How to revert | Replace the list by a splitter with metadata routing. |

### T-07 One preprocessing design for every model

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Every model shares the same `ColumnTransformer`: one-hot fuel (one column, since there are two fuels), standardized numeric inputs and fiber flag, and `po2` computed with `FunctionTransformer(numpy.prod)` and then standardized. `logreg_spline` replaces the linear terms of `o2`, `d0_mm`, `co2` and `he` with cubic B-splines with 4 uniform knots. |
| Options considered | A separate, unscaled preprocessing for gradient boosting; native categorical support in gradient boosting. |
| Reason | One design is simpler to explain and to test, and scaling does not change a tree model. |
| Evidence | `src/flame/training/features.py`; the pipeline loads without `flame` (D-09). |
| Affected files | `src/flame/training/features.py`, `configs/training/*.toml` (`[features]`, `[models.*.spline]`). |
| How to revert | Change `build_preprocessor`. |

### T-08 Fixed hyperparameter values

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Values written in the TOML before the first evaluation and not changed afterwards. Logistic regression: `C = 1.0`, `max_iter = 1000`. Gradient boosting: `max_depth = 3`, `max_iter = 150`, `learning_rate = 0.05`, `min_samples_leaf = 15`, `l2_regularization = 1.0`. Calibration: sigmoid with 3 grouped folds. Ridge: `alpha = 1.0`. |
| Options considered | A hyperparameter search (excluded by D-06). |
| Reason | D-06 asks for conservative, fixed values. These are the library defaults or more regularized than the defaults. |
| Evidence | `configs/training/flex_outcome.toml`, `configs/training/flex_extinction.toml`. |
| Affected files | The two configs. |
| How to revert | Edit the values and record the change as a new decision, because it would be made after seeing results. |

### T-09 Decision threshold of 0.5

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | The outcome is `extinction` when its probability is at least `target.decision_threshold = 0.5`. The same threshold is used for macro-F1 and recall, and is written to `model_info.json`. |
| Options considered | A tuned threshold. |
| Reason | With two classes this equals choosing the more probable class, and it needs no tuning. |
| Evidence | `tests/training/test_selection.py::test_classification_metrics_follow_the_decision_threshold`. |
| Affected files | `configs/training/flex_outcome.toml`, `src/flame/training/evaluate.py` (`predicted_labels`), `src/flame/training/inference.py` (`predict_outputs`). |
| How to revert | Change `target.decision_threshold`. |

### T-10 Details of the support index

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Each fuel has its own scaler, neighbor index and threshold. `in_tested_range` is true only when the nearest test is within the joint threshold **and** no input is outside its own range. The artifact stores the index as a plain dictionary of scikit-learn objects, plus `support_reference.csv` with one line per training row. |
| Options considered | One scaler for both fuels; a joint check that ignores the per-input ranges. |
| Reason | A per-fuel scaler keeps each index self-contained. An input outside its own tested range must never be reported as tested, even when it lies just beyond the range. |
| Evidence | `tests/training/test_support.py`; thresholds per fuel in the outcome model card. |
| Affected files | `src/flame/training/support.py`, `src/flame/training/inference.py`. |
| How to revert | Fit one scaler in `fit_support`. |

### T-11 Published ranges are rounded outward to two decimals

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | The minimum of each numeric input is rounded down and the maximum up, to `interface.range_decimals = 2`, using decimal arithmetic. |
| Options considered | Publishing the raw minimum and maximum. |
| Reason | The lowest tested pressure is slightly above 0.69 atm. With raw limits, a user asking for the nominal Moon pressure of 0.69 atm would be told the pressure is out of range. |
| Evidence | Pressure range 0.69 to 1.02 atm (outcome model card, "Inputs and tested ranges"); `tests/training/test_export.py::test_ranges_are_rounded_outward_without_float_artifacts`. |
| Affected files | `src/flame/training/export.py` (`round_outward`, `tested_range`), `configs/training/*.toml` (`interface.range_decimals`). |
| How to revert | Raise `range_decimals`. |

### T-12 Which inputs raise an error and which are only flagged

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | `reference_predict` raises `ValueError` for a missing or unexpected input, an unknown fuel or suppressant, a value that is not a finite number, a fraction above 0 with suppressant `none`, and a mixture above 1. A number outside its tested range is not an error: the prediction is returned with the input listed in `out_of_range_fields` and confidence `low`. |
| Options considered | Rejecting every out-of-range number. |
| Reason | The architecture wants the simulator to warn and the chatbot to qualify its answer, which needs a response. Structurally wrong requests have no meaningful answer. Hard limits remain the job of the API schema in `ml/`. |
| Evidence | `tests/training/test_inference.py`. |
| Affected files | `src/flame/training/inference.py` (`validate_inputs`, `check_constraints`). |
| How to revert | Raise in `validate_inputs` when `out_of_range` is not empty. |

### T-13 Shape of `training_source` in the response

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | The response carries `training_source` as an object with `dataset`, `accession`, `sha256` and `rows`. `model_info.json` holds the full record (class counts, exclusions, groups). |
| Options considered | The free-text string of the draft schema in `ARCHITECTURE.md` §5. |
| Reason | `/health` must be able to compare the data hash of the model with the hash of the table the chatbot queries. |
| Evidence | `tests/training/test_inference.py::test_response_has_the_contract_fields_and_types`. Part of O-01. |
| Affected files | `src/flame/training/inference.py` (`reference_predict`), `src/flame/training/export.py` (`describe_training_source`). |
| How to revert | Format a string in `reference_predict`. |

### T-14 How the two targets are compared

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | Both targets are compared with the algorithm selected for the reference target. ROC-AUC is computed per repeat on the pooled out-of-fold predictions of the shared rows and then averaged. Spearman correlation is the Pearson correlation of average ranks. |
| Options considered | Comparing every algorithm; importing `scipy.stats.spearmanr`. |
| Reason | Using one algorithm isolates the effect of the target. On the shared rows both targets have the same labels, so the ROC-AUC values are comparable. scipy is not an allowed direct dependency. |
| Evidence | `tests/training/test_sensitivity.py`; sensitivity report. |
| Affected files | `src/flame/training/sensitivity.py`. |
| How to revert | Change `compare_targets_from_data`. |

### T-15 Generated reports carry no timestamp; reference queries live in the config

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | The evaluation report, the sensitivity report and the model card contain no date, so two runs give identical files. The creation date is only in `model_info.json`. The reference queries (the smoke tests) are written in `[[report.reference_queries]]` and their answers are printed in the model card. |
| Options considered | Running the smoke tests by hand and copying the results. |
| Reason | Every number in a report must come from the code, and identical reports are the simplest proof of determinism. |
| Evidence | Outcome model card, "Reference queries"; `reports/code_walkthrough.md`, "Verification", checks 4 and 6. |
| Affected files | `src/flame/training/reporting.py`, `src/flame/training/export.py` (`run_reference_queries`), `configs/training/*.toml` (`[report]`). |
| How to revert | Remove the `[[report.reference_queries]]` entries. |

### T-16 Regression details for `flex-extinction`

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | For the regression task, `StratifiedGroupKFold` stratifies on 4 quantile bins of the target (`cv.stratify_bins`). Predictions are clipped to `target.prediction_bounds = [0, 1]` in the evaluation and in `reference_predict`; the bounds are written to `model_info.json` because the saved pipeline itself does not clip. The response gives the ratio and the diameter in millimetres (ratio x `droplet_d0_mm`). Confidence uses the support only, since there is no class probability. |
| Options considered | `GroupKFold` without stratification; no clipping; a clipping step inside the pipeline. |
| Reason | D-08 asks for the same cross-validation as D-05, and that splitter needs discrete strata. The ratio cannot be below 0 or above 1 physically, and a linear model can leave that interval. Clipping inside the pipeline would need a custom class (D-09). |
| Evidence | `tests/training/test_regression.py`; extinction model card. |
| Affected files | `configs/training/flex_extinction.toml`, `src/flame/training/splits.py` (`stratification_labels`), `src/flame/training/models.py` (`predict_values`), `src/flame/training/inference.py` (`predict_outputs`), `src/flame/training/support.py` (`confidence_level`). |
| How to revert | Edit `[cv].stratify_bins` or remove `target.prediction_bounds`. |

### T-17 Tests put `src` on the path themselves

| Field | Content |
|---|---|
| Date | 2026-11-14 |
| Status | TECHNICAL |
| Decision | `tests/training/conftest.py` inserts `src/` into `sys.path`, as `main.py` does. Unit tests use a synthetic table built in the same file. Integration tests are skipped when `data/processed/flex_tests.csv` is missing. |
| Options considered | A `pyproject.toml` or `pytest.ini` (outside the allowed files). |
| Reason | No packaging file may be added, and `data/` is not committed, so a clone without data must still pass the unit tests. |
| Evidence | `tests/training/`. |
| Affected files | `tests/training/conftest.py`. |
| How to revert | Add packaging metadata and remove the path insertion. |

## Results of D-08

`flex-extinction` 0.2 was built after `flex-outcome` 0.1 passed its verification. All figures come from the
extinction evaluation and the extinction model card.

| Item | Result | Source |
|---|---|---|
| Training rows | 156 of 274, in 40 atmosphere groups. Besides the D-03 filters, 80 rows are not extinctions and 15 extinctions have no measured diameter | extinction evaluation, run table and "Rows excluded" |
| Target | `dext_over_d0` from 0.2483 to 1.0030, mean 0.6942 | extinction evaluation, run table |
| Baseline (`ridge`) | MAE 0.1080, RMSE 0.1407, R2 0.5438 | extinction evaluation, pooled metrics |
| Candidate (`hgb`) | MAE 0.0998, RMSE 0.1317, R2 0.6002 | extinction evaluation, pooled metrics |
| Rule | Mean paired MAE difference -0.0082; the candidate is better in 9 of 10 repeats, so **`hgb` is published** | extinction evaluation, "Paired comparison with the baseline" |
| Error by region | `hgb` is better at about 1 atm and worse than `ridge` at about 0.7 atm with CO2 (0.0907 vs. 0.0775) and with He (0.1357 vs. 0.1150) | extinction evaluation, "Mean absolute error by region" |
| Tested ranges | O2 0.12 to 0.30 and D0 1.49 to 4.67 mm: no flame went out above O2 0.30 | extinction model card, "Inputs and tested ranges" |
| Reference queries | ISS-like air: ratio 0.445 (1.336 mm), in range, `support_n` 11, confidence high. Moon-like atmosphere without suppressant: out of range (`o2`), confidence low, because this model has no rows there | extinction model card, "Reference queries" |

The selection rule looks at the overall MAE only, so the regional weakness at about 0.7 atm does not change
the selection. It is reported here so the team knows where the published model is less accurate.

## Open items

These are not resolved here.

| ID | Item |
|---|---|
| O-01 | **`/predict` contract.** The model returns a binary `outcome` (`extinction` / `no_extinction`), while the draft schema in `ARCHITECTURE.md` §5 has a three-value literal and a required `burning_rate_mm2_s`, which no model predicts. `training_source` is an object here and a string in the draft (T-13). `shared/schemas.py` must be agreed with the frontend and MCP owners. |
| O-02 | **Handoff to `ml/`.** `artifacts/` is ignored by git, so the way the artifact reaches the ML service is undecided. `ml/` must use the scikit-learn version recorded in `model_info.json`. |
| O-03 | **Monotonic constraints** for CO2 and He are not applied. The physical directions must be confirmed in the FLEX papers first. |
| O-04 | **`CLAUDE.md` §3, §10 and §11.8 are out of date.** The Moon smoke test there expects low confidence; without suppressant that no longer applies, because the source table has 17 Moon-like tests (outcome model card, coverage facts). The untested case is the Moon-like atmosphere with a suppressant. |
| O-05 | **BASS-II** waits for verified outcome labels. No BASS-II code exists. |
| O-06 | **Unused information**: the imposed slow flow (mentioned in comments only) and the 24 extinction diameters that appear only in the TestSheets. |
| O-07 | **Probabilities of the published baseline.** D-06 gives the logistic regression balanced class weights and calibrates only the gradient-boosting candidate. Balanced weights make the probabilities describe a balanced population, not the observed class frequencies, and the chatbot will quote them. Calibrating the baseline, or dropping the weights, would change D-06 and was not done. |
| O-08 | **Pressures between the two tested levels.** Following D-07, a pressure near one tested level is reported as in range, although only two levels were tested. Halfway between them the query is flagged (outcome model card, reference query `between_pressure_levels`). The simulator could restrict pressure to the two tested levels. |
| O-09 | **Not verified or not touched.** `environment.yml` pins the installed versions, but the environment was not created, because installing was out of bounds. `main.py` still has comments and a docstring; it was outside the files allowed to change. |
