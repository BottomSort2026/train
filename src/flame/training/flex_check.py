from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from flame.training import flex_data

REPORT = flex_data.TRAIN_ROOT / "reports" / "flex_data_check.md"

MOON = {"p_min": 0.65, "p_max": 0.75, "o2_min": 0.295}
ISS = {"p_min": 0.95, "p_max": 1.05, "o2_min": 0.195, "o2_max": 0.225}
OUTCOMES = ["extinction", "disruption", "completion"]


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    df = df.reset_index().rename(columns={"index": ""}) if index else df
    header = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join("" if pd.isna(v) else (f"{v:.2f}" if isinstance(v, float) else str(v))
                              for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, rule, *rows])


def by_outcome(mask: pd.Series, df: pd.DataFrame) -> dict:
    return {o: int(mask[df["test_end"] == o].sum()) for o in OUTCOMES}


def in_moon_box(df: pd.DataFrame) -> pd.Series:
    return df["pressure_atm"].between(MOON["p_min"], MOON["p_max"]) & (df["o2"] >= MOON["o2_min"])


def in_iss_box(df: pd.DataFrame) -> pd.Series:
    return (df["pressure_atm"].between(ISS["p_min"], ISS["p_max"]) & df["o2"].between(ISS["o2_min"], ISS["o2_max"])
            & (df["suppressant"] == "none"))


def raw_token_kind(text: str) -> str:
    if text == "":
        return "blank"
    if text in {"–", "-"}:
        return "dash"
    try:
        return "zero" if float(text.lstrip("~")) == 0 else "number"
    except ValueError:
        return f"other ({text})"


def section_outcomes(df: pd.DataFrame) -> str:
    counts = df["test_end"].value_counts().reindex(OUTCOMES)
    table = pd.crosstab([df["fuel"], df["suppressant"]], df["test_end"], margins=True)[OUTCOMES + ["All"]]
    return "\n\n".join([
        "## 1. Outcomes (test end)",
        f"Counts: {counts.to_dict()} (total {len(df)}).",
        md_table(table),
    ])


def section_disruption(df: pd.DataFrame, sheets: pd.DataFrame) -> str:
    flags = [c for c in df.columns if c.startswith("mentions_")]
    mention = pd.DataFrame({f: by_outcome(df[f], df) for f in flags}).T
    mention.columns = [f"{c} (n={int((df['test_end'] == c).sum())})" for c in mention.columns]
    high_o2 = df["o2"] >= MOON["o2_min"]
    o2_share = pd.DataFrame({
        "tests": df["test_end"].value_counts().reindex(OUTCOMES),
        "with O2 >= 0.30": [int((high_o2 & (df["test_end"] == o)).sum()) for o in OUTCOMES],
        "fiber-supported": [int((df["fiber"].eq(True) & (df["test_end"] == o)).sum()) for o in OUTCOMES],
        "median D/D0 at end": [df.loc[df["test_end"] == o, "dext_over_d0"].median() for o in OUTCOMES],
    })
    disrupted = df[df["test_end"] == "disruption"].merge(sheets[["testsheet_id", "comment"]], on="testsheet_id")
    examples = []
    for _, row in disrupted.head(60).iterrows():
        sentences = [s for s in re.split(r"(?<=\.)\s+", row["comment"]) if re.search("disrupt", s, re.I)]
        if sentences and len(examples) < 8:
            examples.append(f"- `{row['cir_id']}` ({row['fuel']}, O2 {row['o2']:.2f}, "
                            f"fiber {row['fiber']}): {sentences[0]}")
    return "\n\n".join([
        "## 2. What \"Disruption\" means",
        "Comment keywords by test end (number of tests whose TestSheet comment mentions the topic):",
        md_table(mention),
        "Where disruptions happen:",
        md_table(o2_share),
        "Examples from the TestSheets:",
        "\n".join(examples),
    ])


def section_extinction_diameter(df: pd.DataFrame, raw: pd.DataFrame) -> str:
    kinds = raw["dext_mm"].map(raw_token_kind)
    table = pd.crosstab(raw["test_end"].str.lower(), kinds).reindex(OUTCOMES)
    non_blank = int((raw["dext_mm"] != "").sum())
    return "\n\n".join([
        "## 3. Extinction diameters: why 225 values vs. 186 extinctions",
        f"The raw column has {non_blank} non-blank cells, but they include dashes ('–' = not measured) and zeros:",
        md_table(table),
        f"Real numeric extinction diameters: {int((kinds == 'number').sum())} "
        f"({int(((kinds == 'number') & (raw['test_end'] == 'Extinction')).sum())} on Extinction tests).",
        "D_ext/D0 on Extinction tests: "
        + json.dumps(df.loc[df["test_end"] == "extinction", "dext_over_d0"].describe().round(2).to_dict()),
    ])


def section_missing(df: pd.DataFrame) -> str:
    cols = ["pressure_atm", "d0_mm", "dext_mm", "burning_rate_mm2_s", "burn_time_s", "fiber", "temperature_k"]
    table = pd.DataFrame({c: by_outcome(df[c].isna(), df) for c in cols}).T
    return "\n\n".join(["## 4. Missing values by test end", md_table(table)])


def section_coverage(df: pd.DataFrame) -> str:
    level = pd.cut(df["pressure_atm"], [0, 0.85, 1.5, 2.5, 3.2], labels=["~0.7 atm", "~1 atm", "~2 atm", "~3 atm"])
    by_level = pd.crosstab([level, df["suppressant"]], df["test_end"])[OUTCOMES]
    o2_range = df.groupby([level, "fuel"], observed=True)["o2"].agg(["min", "max", "count"])
    moon, iss = in_moon_box(df), in_iss_box(df)
    moon_rows = df.loc[moon, ["cir_id", "fuel", "pressure_atm", "o2", "suppressant", "fiber", "d0_mm", "test_end"]]
    high = df.loc[df["high_pressure"], ["suppressant", "o2", "suppressant_fraction"]].drop_duplicates()
    return "\n\n".join([
        "## 5. Coverage of the scenarios",
        "Tests per pressure level and suppressant:",
        md_table(by_level),
        "O2 range per pressure level and fuel:",
        md_table(o2_range),
        f"Above {flex_data.HIGH_PRESSURE_ATM} atm there are {int(df['high_pressure'].sum())} tests, all with "
        f"these mixtures: {high.to_dict('records')}. Pressure above 1 atm is confounded with 70 % CO2.",
        f"**Moon-like** (≈0.7 atm, O2 ≥ 0.30): {int(moon.sum())} tests, outcomes {by_outcome(moon, df)}.",
        md_table(moon_rows, index=False),
        f"**ISS-like** (≈1 atm, O2 0.20–0.22, no suppressant): {int(iss.sum())} tests, outcomes {by_outcome(iss, df)}.",
    ])


def section_hidden_variables(df: pd.DataFrame) -> str:
    heptane_ext = (df["fuel"] == "heptane") & (df["test_end"] == "extinction")
    fiber = pd.crosstab([df["fuel"], df["fiber"].map({True: "fiber", False: "free"}).fillna("unknown")],
                        df["test_end"])[OUTCOMES]
    return "\n\n".join([
        "## 6. Variables that are not in the table",
        "Support fiber (TestSheet field `Fiber`): fiber-supported droplets lose heat to the fiber.",
        md_table(fiber),
        f"Comments mentioning translation (an imposed slow flow): {int(df['mentions_translation'].sum())} tests "
        f"{by_outcome(df['mentions_translation'], df)}.",
        f"Heptane extinctions: {int(heptane_ext.sum())}; their comments mention cool flames in "
        f"{int((heptane_ext & df['mentions_cool_flame']).sum())}. After the visible flame goes out, heptane can keep "
        "burning as a dim cool flame, so 'extinction' means *visible* flame extinction.",
        f"Calibration tests (CAL in the ID): {int(df['is_calibration'].sum())}.",
        f"Gas temperature: {df['temperature_k'].min():.1f}–{df['temperature_k'].max():.1f} K "
        "(nearly constant, not an input).",
    ])


def common_exclusions(df: pd.DataFrame) -> pd.Series:
    return (df["duplicate_of"].notna() | df["d0_mm"].isna() | df["pressure_atm"].isna() | df["high_pressure"])


def section_training_sets(df: pd.DataFrame) -> str:
    excluded = common_exclusions(df)
    base = df[~excluded]
    moon = in_moon_box(base)
    options = {
        "A. 3 classes as published": base,
        "B. Drop Disruption (extinction vs completion)": base[base["test_end"] != "disruption"],
        "C. Disruption = flame not extinguished by the atmosphere": base,
    }
    rows = []
    for name, data in options.items():
        target = data["test_end"]
        if name.startswith("C"):
            target = target.replace({"disruption": "not extinguished", "completion": "not extinguished"})
        elif name.startswith("B"):
            target = target.replace({"completion": "not extinguished"})
        m = in_moon_box(data)
        rows.append({
            "option": name,
            "rows": len(data),
            "classes": target.value_counts().to_dict(),
            "atmosphere groups": data["atmosphere_id"].nunique(),
            "Moon-like rows": f"{int(m.sum())} {target[m].value_counts().to_dict()}",
        })
    groups = base.groupby("atmosphere_id")["test_end"].agg(["size", lambda s: s.nunique() > 1])
    groups.columns = ["size", "mixed"]
    return "\n\n".join([
        "## 7. Candidate training sets for the outcome model",
        f"Common exclusions ({int(excluded.sum())} rows): duplicates {int(df['duplicate_of'].notna().sum())}, "
        f"missing D0 {int(df['d0_mm'].isna().sum())}, missing pressure {int(df['pressure_atm'].isna().sum())}, "
        f"above {flex_data.HIGH_PRESSURE_ATM} atm {int(df['high_pressure'].sum())} (overlaps counted once).",
        md_table(pd.DataFrame(rows), index=False),
        f"CV groups (`atmosphere_id`, one chamber fill = same pressure level and gas mixture): {len(groups)} groups, "
        f"sizes {groups['size'].min()}–{groups['size'].max()} (median {groups['size'].median():.0f}); "
        f"{int(groups['mixed'].sum())} groups contain more than one outcome. Moon-like tests sit in "
        f"{base.loc[moon, 'atmosphere_id'].nunique()} groups.",
    ])


def run_check(report: Path = REPORT) -> Path:
    df = flex_data.load_flex()
    manifest = json.loads(flex_data.MANIFEST.read_text())
    raw = flex_data.read_raw_table()
    sheets = flex_data.read_testsheets()
    parts = [
        "# FLEX (PSI-69) data check",
        f"Generated by `python main.py train check-flex` from `{manifest['processed_file']}` "
        f"(sha256 `{manifest['sha256'][:12]}…`, {manifest['rows']} rows, {manifest['source_files']['testsheets']} "
        "TestSheets).",
        "### Cleaning log\n" + "\n".join(f"- {line}" for line in manifest["cleaning_log"]),
        section_outcomes(df),
        section_disruption(df, sheets),
        section_extinction_diameter(df, raw),
        section_missing(df),
        section_coverage(df),
        section_hidden_variables(df),
        section_training_sets(df),
    ]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n\n".join(parts) + "\n")
    return report
