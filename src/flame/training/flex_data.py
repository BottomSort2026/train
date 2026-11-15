from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ACCESSION = "PSI-69"
TRAIN_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = TRAIN_ROOT / "data" / "raw" / ACCESSION
RAW_TABLE = RAW_DIR / "PSI-69_Experimental table_FLEX.csv"
TESTSHEET_DIR = RAW_DIR / "TestSheets"
PROCESSED_CSV = TRAIN_ROOT / "data" / "processed" / "flex_tests.csv"
MANIFEST = PROCESSED_CSV.with_suffix(".json")

MMHG_PER_ATM = 760.0
ENCODING = "cp1252"

RAW_COLUMNS = {
    "FLEX Test #": "table_row",
    "FLEX Identifier": "cir_id",
    "Test Date": "test_date",
    "Test GMT": "test_gmt",
    "Fuel": "fuel",
    "Ambient pressure; mmHg": "pressure_mmhg",
    "O initial ambient composition; mole fraction": "o2",
    "N initial ambient composition; mole fraction": "n2",
    "CO initial ambient composition; mole fraction": "co2",
    "He initial ambient composition; mole fraction": "he",
    "Droplet initial diameter; mm": "d0_mm",
    "Visible flame extinction diameter; mm": "dext_mm",
    "Burning rate; mm": "burning_rate_mm2_s",
    "Burn time; s": "burn_time_s",
    "Test end": "test_end",
}
NUMERIC_COLUMNS = ["pressure_mmhg", "o2", "n2", "co2", "he", "d0_mm", "dext_mm",
                   "burning_rate_mm2_s", "burn_time_s"]

COMMENT_FLAGS = {
    "mentions_disruption": r"disrupt",
    "mentions_cool_flame": r"cool.flame",
    "mentions_translation": r"translat",
    "mentions_fiber_oscillation": r"oscillat",
    "mentions_gas_bubble": r"bubble",
    "mentions_out_of_view": r"out of (?:the )?(?:HiBMS )?FOV|leaves? the (?:HiBMS )?FOV",
    "mentions_missing_images": r"did not record|stopped recording",
    "mentions_hits_igniter": r"hits? the (?:upper |lower )?igniter",
}

HIGH_PRESSURE_ATM = 1.5
COMPOSITION_TOLERANCE = 0.015

PSI_API = "https://psi.nasa.gov/geode-py/ws"
_HEADERS = {"User-Agent": "flame-train"}


def _psi_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _download_file(file_name: str, dest: Path, attempts: int = 4) -> None:
    query = urllib.parse.urlencode({"file": file_name, "redirect": "false"})
    for attempt in range(1, attempts + 1):
        try:
            signed_url = _psi_get(f"{PSI_API}/studies/{ACCESSION}/download?{query}").decode().strip().strip('"')
            dest.write_bytes(_psi_get(signed_url))
            return
        except OSError:
            if attempt == attempts:
                raise


def _rename(current: dict[str, str], old_name: str, new_name: str) -> None:
    matches = [path for path, name in current.items() if name == old_name]
    for path in matches:
        del current[path]
        current[str(PurePosixPath(path).with_name(new_name))] = new_name


def current_file_names(versions: list[dict]) -> list[str]:
    current: dict[str, str] = {}
    for version in sorted(versions, key=lambda v: v["version"]):
        for f in version.get("added", []) + version.get("modified", []):
            if "remote_path" in f:
                current[f["remote_path"]] = f["file_name"]
            else:
                _rename(current, f["file_name"], f["new_file_name"])
        for f in version.get("removed", []):
            current.pop(f["remote_path"], None)
    return sorted(current.values())


def _current_files() -> list[str]:
    versions = json.loads(_psi_get(f"{PSI_API}/repo/investigations/{ACCESSION}/versions"))["versions"]
    return current_file_names(versions)


def download_raw(force: bool = False) -> None:
    TESTSHEET_DIR.mkdir(parents=True, exist_ok=True)
    names = _current_files()
    wanted = [(RAW_TABLE.name, RAW_TABLE)]
    wanted += [(n, TESTSHEET_DIR / n) for n in names if n.endswith("_TestSheet.txt")]
    missing = [(n, p) for n, p in wanted if force or not p.exists()]
    for i, (name, dest) in enumerate(missing, 1):
        _download_file(name, dest)
        if i % 25 == 0:
            print(f"  downloaded {i}/{len(missing)}")
    print(f"{len(wanted)} files wanted, {len(missing)} downloaded into {RAW_DIR}")


def read_raw_table(path: Path = RAW_TABLE) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)
    if list(raw.columns) != list(RAW_COLUMNS):
        raise ValueError(f"Unexpected FLEX table header: {list(raw.columns)}")
    return raw.rename(columns=RAW_COLUMNS).apply(lambda col: col.str.strip())


def _read_text(path: Path) -> str:
    return path.read_bytes().decode(ENCODING).replace("\r\n", "\n").replace("\r", "\n")


def _field(text: str, pattern: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else None


def parse_testsheet(path: Path) -> dict:
    text = _read_text(path)
    comment = re.search(r"COMMENTS:\s*(.*)\Z", text, re.S)
    body = text.split("\n", 1)[1] if "\n" in text else text
    return {
        "testsheet_id": path.name.split("_")[0],
        "cir_id": _field(text, r"CIR Test Number:\s*(\S+)"),
        "sheet_pressure_atm": _field(text, r"Pressure \(atm\) =\s*(\S+)"),
        "sheet_temperature_k": _field(text, r"Temperature \(K\) =\s*(\S+)"),
        "sheet_d0_mm": _field(text, r"D0\s*\(mm\) =\s*(\S+)"),
        "sheet_dext_mm": _field(text, r"Dext\s*\(mm\) =\s*(\S+)"),
        "sheet_kavg_mm2_s": _field(text, r"kavg\s*\(mm2/s\) =\s*(\S+)"),
        "fiber": _field(text, r"Fiber:\s*(\w+)"),
        "comment": comment.group(1).strip() if comment else "",
        "_body_hash": hashlib.sha256(body.encode()).hexdigest(),
    }


def read_testsheets(directory: Path = TESTSHEET_DIR) -> pd.DataFrame:
    sheets = pd.DataFrame([parse_testsheet(p) for p in sorted(directory.glob("*_TestSheet.txt"))])
    if sheets.empty:
        raise FileNotFoundError(f"No TestSheets in {directory}. Run: python main.py train fetch-flex")
    for col in ["sheet_pressure_atm", "sheet_temperature_k", "sheet_d0_mm", "sheet_dext_mm", "sheet_kavg_mm2_s"]:
        sheets[col] = pd.to_numeric(sheets[col], errors="coerce")
    sheets["fiber"] = sheets["fiber"].str.lower().map({"yes": True, "no": False})
    copies = sheets["_body_hash"].duplicated(keep="first")
    sheets.attrs["copied_sheets"] = sheets.loc[copies, "testsheet_id"].tolist()
    return sheets.loc[~copies].drop(columns="_body_hash").reset_index(drop=True)


def _to_number(text: pd.Series) -> tuple[pd.Series, pd.Series]:
    approximate = text.str.startswith("~")
    values = pd.to_numeric(text.str.lstrip("~").replace({"": None, "–": None, "-": None}), errors="coerce")
    return values, approximate


def _match_testsheets(table: pd.DataFrame, sheets: pd.DataFrame) -> pd.DataFrame:
    candidates = table[["table_row", "cir_id", "d0_mm"]].merge(sheets, on="cir_id", how="left")
    candidates["_gap"] = (candidates["d0_mm"] - candidates["sheet_d0_mm"]).abs().fillna(np.inf)
    best = candidates.sort_values(["table_row", "_gap"]).drop_duplicates("table_row")
    return best.drop(columns=["cir_id", "d0_mm", "_gap"])


def _impute_missing_pressure(df: pd.DataFrame, log: list[str]) -> None:
    for i in df.index[df["pressure_mmhg"].isna()]:
        row = df.loc[i]
        same = (df["cir_series"] == row["cir_series"]) & df["pressure_mmhg"].notna()
        for col in ["o2", "co2", "he"]:
            same &= (df[col] - row[col]).abs() < 0.011
        if same.any():
            value = float(df.loc[same, "pressure_mmhg"].median())
            df.loc[i, ["pressure_mmhg", "pressure_imputed"]] = [value, True]
            log.append(f"{row['cir_id']}: pressure was 0 (missing); imputed {value:.1f} mmHg from "
                       f"{int(same.sum())} tests of series {row['cir_series']} with the same composition")
        else:
            log.append(f"{row['cir_id']}: pressure missing and could not be imputed")


def clean_flex(table: pd.DataFrame, sheets: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    log: list[str] = []
    df = table.copy()

    df["table_row"] = df["table_row"].astype(int)
    df["fuel"] = df["fuel"].str.lower()
    df["test_end"] = df["test_end"].str.lower()
    df["test_date"] = pd.to_datetime(df["test_date"], format="mixed").dt.date.astype(str)
    df["cir_series"] = df["cir_id"].str.extract(r"^([A-Z]?\d+)", expand=False)
    df["is_calibration"] = df["cir_id"].str.contains("CAL")

    for col in NUMERIC_COLUMNS:
        df[col], approximate = _to_number(df[col])
        if approximate.any():
            df[f"{col}_approx"] = approximate
            log.append(f"{col}: {int(approximate.sum())} approximate values ('~') kept without the '~'")
    zero_pressure = df["pressure_mmhg"] == 0
    df.loc[zero_pressure, "pressure_mmhg"] = np.nan
    df["pressure_imputed"] = False
    _impute_missing_pressure(df, log)

    df["pressure_atm"] = df["pressure_mmhg"] / MMHG_PER_ATM
    df["composition_sum"] = df[["o2", "n2", "co2", "he"]].sum(axis=1)
    df["composition_off"] = (df["composition_sum"] - 1).abs() > COMPOSITION_TOLERANCE
    df["suppressant"] = np.select([df["co2"] > 0, df["he"] > 0], ["co2", "he"], default="none")
    df["suppressant_fraction"] = df["co2"] + df["he"]
    df["dext_over_d0"] = df["dext_mm"] / df["d0_mm"]
    df["high_pressure"] = df["pressure_atm"] > HIGH_PRESSURE_ATM
    df["atmosphere_id"] = (df["pressure_atm"].round(1).map("{:.1f}atm".format) + "_O2-" + df["o2"].map("{:.2f}".format)
                           + "_CO2-" + df["co2"].map("{:.2f}".format) + "_He-" + df["he"].map("{:.2f}".format))

    key = df[["cir_id", "test_date", "test_gmt"]].astype(str).agg("|".join, axis=1)
    first_row = df.groupby(key)["table_row"].transform("min")
    df["duplicate_of"] = first_row.where(first_row != df["table_row"])
    for _, row in df[df["duplicate_of"].notna()].iterrows():
        log.append(f"{row['cir_id']}: table rows {int(row['duplicate_of'])} and {row['table_row']} are the "
                   f"same test (same ID, date and time) with different measurements; flagged duplicate_of")

    df = df.merge(_match_testsheets(df, sheets), on="table_row", how="left")
    df["has_testsheet"] = df["testsheet_id"].notna()
    df["temperature_k"] = df["sheet_temperature_k"].where(df["sheet_temperature_k"] > 0)
    comments = df["comment"].fillna("")
    for flag, pattern in COMMENT_FLAGS.items():
        df[flag] = comments.str.contains(pattern, case=False, regex=True)
    for sheet_id in sheets.attrs.get("copied_sheets", []):
        log.append(f"{sheet_id}: TestSheet is a copy of another sheet's content; ignored")
    for _, row in df[~df["has_testsheet"]].iterrows():
        log.append(f"{row['cir_id']} (table row {row['table_row']}): no TestSheet; fiber and comments unknown")
    _log_disagreements(df, log)

    approx_flags = [c for c in df.columns if c.endswith("_approx")]
    columns = [
        "table_row", "testsheet_id", "cir_id", "cir_series", "test_date", "test_gmt", "fuel",
        "pressure_mmhg", "pressure_atm", "pressure_imputed", "o2", "n2", "co2", "he", "composition_sum",
        "composition_off", "suppressant", "suppressant_fraction", "temperature_k", "fiber",
        "d0_mm", "dext_mm", "dext_over_d0", "burning_rate_mm2_s", "burn_time_s", *approx_flags,
        "test_end", "is_calibration", "high_pressure", "duplicate_of", "has_testsheet", "atmosphere_id",
        *COMMENT_FLAGS,
    ]
    return df[columns].sort_values("table_row").reset_index(drop=True), log


def _log_disagreements(df: pd.DataFrame, log: list[str]) -> None:
    checks = [("d0_mm", "sheet_d0_mm", 0.011), ("dext_mm", "sheet_dext_mm", 0.011),
              ("burning_rate_mm2_s", "sheet_kavg_mm2_s", 0.011)]
    for table_col, sheet_col, tolerance in checks:
        differ = (df[table_col] - df[sheet_col]).abs() > tolerance
        only_sheet = df[table_col].isna() & df[sheet_col].notna()
        if differ.any() or only_sheet.any():
            log.append(f"{table_col}: {int(differ.sum())} rows differ from the TestSheet by more than {tolerance}; "
                       f"{int(only_sheet.sum())} rows are empty in the table but have a TestSheet value (not filled)")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_flex() -> dict:
    table = read_raw_table()
    sheets = read_testsheets()
    clean, log = clean_flex(table, sheets)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_CSV, index=False, float_format="%.6g", lineterminator="\n")
    manifest = {
        "dataset": "flex_tests",
        "accession": ACCESSION,
        "source_files": {RAW_TABLE.name: sha256(RAW_TABLE), "testsheets": int(len(sheets))},
        "processed_file": PROCESSED_CSV.name,
        "sha256": sha256(PROCESSED_CSV),
        "rows": int(len(clean)),
        "created": date.today().isoformat(),
        "pandas_version": pd.__version__,
        "cleaning_log": log,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def load_flex(path: Path = PROCESSED_CSV, verify: bool = True) -> pd.DataFrame:
    if verify:
        expected = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"{path.name} does not match the hash in its manifest. Re-run prepare-flex.")
    return pd.read_csv(path, dtype={"testsheet_id": "string", "duplicate_of": "Int64"})
