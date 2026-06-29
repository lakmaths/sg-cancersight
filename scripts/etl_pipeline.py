"""
SG-CancerSight — Phase 1: ETL Pipeline
=======================================
Loads, validates, cleans, and transforms the SEER breast cancer dataset.
Produces a reproducible, analysis-ready output with full audit trail.

Author : Dr. Lakshmi C.
Dataset: SEER Breast Cancer Registry (synthetic research replica)
JD Map : NCCS Data Analyst (DDOIT) — Data Management & Wrangling requirements
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
RAW_PATH   = BASE_DIR / "data" / "raw"  / "seer_raw.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "seer_clean.csv"
DICT_PATH  = BASE_DIR / "data" / "processed" / "data_dictionary.json"
LOG_PATH   = BASE_DIR / "data" / "processed" / "etl_run.log"

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
VALID_STAGES     = {"I", "IIA", "IIB", "IIIA", "IIIB", "IIIC"}
VALID_T_STAGE    = {"T1", "T2", "T3", "T4"}
VALID_N_STAGE    = {"N0", "N1", "N2", "N3"}
VALID_A_STAGE    = {"Regional", "Distant"}
VALID_STATUS     = {"Alive", "Dead"}
VALID_ER_PR      = {"Positive", "Negative"}
TUMOUR_SIZE_MAX  = 500   # mm — sentinel 999 = unknown
MIN_AGE          = 18
MAX_AGE          = 100
MIN_SURVIVAL     = 1     # months


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — Load
# ══════════════════════════════════════════════════════════════════════════
def load_raw(path: Path) -> pd.DataFrame:
    """Load raw CSV; log shape and dtypes."""
    log.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path)
    log.info(f"Raw shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — Validate (flag issues before removing anything)
# ══════════════════════════════════════════════════════════════════════════
def validate(df: pd.DataFrame) -> dict:
    """
    Run validation checks. Returns a report dict.
    Does NOT modify the dataframe — purely diagnostic.
    """
    report = {}

    # 2a. Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    report["missing"] = missing[missing > 0].to_dict()
    report["missing_pct"] = missing_pct[missing > 0].to_dict()
    log.info(f"Missing values found in: {list(report['missing'].keys())}")

    # 2b. Invalid ages
    bad_age = df[(df["Age"] < MIN_AGE) | (df["Age"] > MAX_AGE)].index.tolist()
    report["invalid_age_rows"] = bad_age
    log.warning(f"Invalid ages (< {MIN_AGE} or > {MAX_AGE}): {len(bad_age)} rows")

    # 2c. Negative or zero survival months
    bad_surv = df[df["Survival Months"] < MIN_SURVIVAL].index.tolist()
    report["invalid_survival_rows"] = bad_surv
    log.warning(f"Invalid survival months (< {MIN_SURVIVAL}): {len(bad_surv)} rows")

    # 2d. Tumour size sentinel (999 = unknown in SEER convention)
    sentinel_size = df[df["Tumor Size"] == 999].index.tolist()
    report["sentinel_tumor_size_rows"] = sentinel_size
    log.warning(f"Tumour size sentinel (999): {len(sentinel_size)} rows")

    # 2e. Unexpected category values
    for col, valid_set in [
        ("6th Stage",           VALID_STAGES),
        ("T Stage",             VALID_T_STAGE),
        ("N Stage",             VALID_N_STAGE),
        ("A Stage",             VALID_A_STAGE),
        ("Status",              VALID_STATUS),
        ("Estrogen Status",     VALID_ER_PR),
        ("Progesterone Status", VALID_ER_PR),
    ]:
        if col in df.columns:
            unexpected = df[~df[col].isin(valid_set)][col].unique().tolist()
            if unexpected:
                report[f"unexpected_{col}"] = unexpected
                log.warning(f"Unexpected values in '{col}': {unexpected}")

    # 2f. Nodes positive > nodes examined (logical inconsistency)
    if "Reginol Node Positive" in df.columns and "Regional Node Examined" in df.columns:
        bad_nodes = df[
            df["Reginol Node Positive"].notna() &
            df["Regional Node Examined"].notna() &
            (df["Reginol Node Positive"] > df["Regional Node Examined"])
        ].index.tolist()
        report["nodes_positive_gt_examined"] = bad_nodes
        log.warning(f"Nodes positive > nodes examined: {len(bad_nodes)} rows")

    log.info("Validation complete.")
    return report


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — Clean
# ══════════════════════════════════════════════════════════════════════════
def clean(df: pd.DataFrame, validation_report: dict) -> pd.DataFrame:
    """
    Apply documented cleaning rules.
    Every decision is logged so the audit trail is reproducible.
    """
    df = df.copy()
    n_start = len(df)
    log.info(f"Starting cleaning: {n_start:,} rows")

    # 3a. Rename column with typo (matches real SEER dataset)
    df.rename(columns={"Reginol Node Positive": "Regional Node Positive"}, inplace=True)
    log.info("Renamed 'Reginol Node Positive' → 'Regional Node Positive'")

    # 3b. Remove invalid ages
    bad_age_rows = validation_report.get("invalid_age_rows", [])
    df.drop(index=bad_age_rows, inplace=True)
    log.info(f"Removed {len(bad_age_rows)} rows with invalid age")

    # 3c. Remove negative survival months
    bad_surv_rows = validation_report.get("invalid_survival_rows", [])
    df.drop(index=bad_surv_rows, inplace=True)
    log.info(f"Removed {len(bad_surv_rows)} rows with negative survival months")

    # 3d. Recode tumour size sentinel 999 → NaN
    sentinel_rows = validation_report.get("sentinel_tumor_size_rows", [])
    df.loc[df.index.isin(sentinel_rows), "Tumor Size"] = np.nan
    log.info(f"Recoded {len(sentinel_rows)} tumour size sentinel (999) → NaN")

    # 3e. Impute missing tumour size with median within T Stage group
    before_missing = df["Tumor Size"].isna().sum()
    df["Tumor Size"] = df.groupby("T Stage")["Tumor Size"].transform(
        lambda x: x.fillna(x.median())
    )
    after_missing = df["Tumor Size"].isna().sum()
    log.info(f"Imputed Tumor Size: {before_missing} → {after_missing} missing (median within T Stage)")

    # 3f. Impute missing Regional Node Examined with median
    df["Regional Node Examined"] = df["Regional Node Examined"].fillna(
        df["Regional Node Examined"].median()
    ).round().astype("Int64")
    log.info("Imputed Regional Node Examined with overall median")

    # 3g. Impute missing Grade with mode
    mode_grade = df["Grade"].mode()[0]
    df["Grade"] = df["Grade"].fillna(mode_grade)
    log.info(f"Imputed Grade with mode: '{mode_grade}'")

    # 3h. Standardise string columns (strip whitespace, consistent case)
    str_cols = ["Race", "Marital Status", "T Stage", "N Stage", "6th Stage",
                "Grade", "A Stage", "Estrogen Status", "Progesterone Status",
                "Status"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3i. Encode binary outcome: Status → numeric (1=Dead, 0=Alive)
    df["Status_Code"] = (df["Status"] == "Dead").astype(int)
    log.info("Encoded Status → Status_Code (1=Dead, 0=Alive)")

    # 3j. Create analysis-ready age groups
    bins   = [0, 39, 49, 59, 69, 150]
    labels = ["<40", "40–49", "50–59", "60–69", "70+"]
    df["Age Group"] = pd.cut(df["Age"], bins=bins, labels=labels, right=True)
    log.info("Created Age Group categorical variable")

    # 3k. Create simplified stage grouping for analysis
    stage_map = {
        "I":    "Stage I",
        "IIA":  "Stage II",
        "IIB":  "Stage II",
        "IIIA": "Stage III",
        "IIIB": "Stage III",
        "IIIC": "Stage III",
    }
    df["Stage Group"] = df["6th Stage"].map(stage_map).fillna("Unknown")
    log.info("Created Stage Group (I / II / III)")

    # 3l. Create ER/PR combined receptor status
    df["Receptor Status"] = np.where(
        (df["Estrogen Status"] == "Positive") | (df["Progesterone Status"] == "Positive"),
        "Hormone Receptor+",
        "Triple Negative / HR-"
    )
    log.info("Created Receptor Status variable")

    # 3l-fix. Cap nodes positive at nodes examined (logical constraint)
    both = df["Regional Node Positive"].notna() & df["Regional Node Examined"].notna()
    violations = (df.loc[both, "Regional Node Positive"] > df.loc[both, "Regional Node Examined"]).sum()
    df.loc[both, "Regional Node Positive"] = df.loc[both].apply(
        lambda r: min(r["Regional Node Positive"], r["Regional Node Examined"]), axis=1
    )
    log.info(f"Capped Regional Node Positive to ≤ Regional Node Examined for {violations} rows")

    # 3m. Reset index after row removals
    df.reset_index(drop=True, inplace=True)

    n_end = len(df)
    log.info(f"Cleaning complete: {n_start:,} → {n_end:,} rows ({n_start - n_end} removed)")
    return df


# ══════════════════════════════════════════════════════════════════════════
# STEP 4 — Data dictionary
# ══════════════════════════════════════════════════════════════════════════
def build_data_dictionary(df: pd.DataFrame) -> dict:
    """
    Auto-generate a data dictionary documenting every column:
    type, missing count/%, allowed values (for categoricals), range (for numerics).
    """
    dd = {}
    for col in df.columns:
        entry = {
            "dtype":        str(df[col].dtype),
            "n_missing":    int(df[col].isna().sum()),
            "pct_missing":  round(df[col].isna().sum() / len(df) * 100, 2),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            entry["min"]  = float(df[col].min()) if not df[col].isna().all() else None
            entry["max"]  = float(df[col].max()) if not df[col].isna().all() else None
            entry["mean"] = round(float(df[col].mean()), 2) if not df[col].isna().all() else None
        else:
            entry["unique_values"] = sorted(df[col].dropna().unique().tolist())
            entry["n_unique"]      = int(df[col].nunique())
        dd[col] = entry

    # Add human-readable descriptions
    descriptions = {
        "Age":                    "Age at diagnosis (years)",
        "Race":                   "Race/ethnicity category",
        "Marital Status":         "Marital status at diagnosis",
        "T Stage":                "Tumour size stage (AJCC 6th edition)",
        "N Stage":                "Regional lymph node involvement",
        "6th Stage":              "Overall AJCC 6th edition stage",
        "Grade":                  "Histological tumour grade",
        "A Stage":                "Regional vs distant spread",
        "Tumor Size":             "Tumour size in mm",
        "Regional Node Examined": "Number of lymph nodes examined",
        "Regional Node Positive": "Number of positive lymph nodes",
        "Estrogen Status":        "Oestrogen receptor status",
        "Progesterone Status":    "Progesterone receptor status",
        "Survival Months":        "Months from diagnosis to last follow-up or death",
        "Status":                 "Vital status at last follow-up (Alive/Dead)",
        "Status_Code":            "Numeric outcome: 1=Dead (event), 0=Alive (censored)",
        "Age Group":              "Derived: 10-year age bands for analysis",
        "Stage Group":            "Derived: simplified stage grouping (I/II/III)",
        "Receptor Status":        "Derived: hormone receptor positive vs negative",
    }
    for col, desc in descriptions.items():
        if col in dd:
            dd[col]["description"] = desc

    return dd


# ══════════════════════════════════════════════════════════════════════════
# STEP 5 — Quality report summary
# ══════════════════════════════════════════════════════════════════════════
def quality_summary(df_raw: pd.DataFrame, df_clean: pd.DataFrame,
                    val_report: dict) -> None:
    """Print a concise quality report to console."""
    print("\n" + "="*60)
    print("  SG-CancerSight — Phase 1 Quality Report")
    print("="*60)
    print(f"  Raw records    : {len(df_raw):>6,}")
    print(f"  Clean records  : {len(df_clean):>6,}")
    print(f"  Rows removed   : {len(df_raw)-len(df_clean):>6,}")
    print(f"  Columns        : {df_clean.shape[1]:>6}")
    print()
    print("  Issues found and resolved:")
    print(f"    Invalid ages          : {len(val_report.get('invalid_age_rows', []))}")
    print(f"    Negative survival     : {len(val_report.get('invalid_survival_rows', []))}")
    print(f"    Tumour size sentinels : {len(val_report.get('sentinel_tumor_size_rows', []))}")
    miss = val_report.get("missing", {})
    for col, cnt in miss.items():
        pct = val_report["missing_pct"].get(col, 0)
        print(f"    Missing {col:<28}: {cnt:>3} ({pct}%) → imputed")
    print()
    print("  Derived variables created:")
    print("    Age Group, Stage Group, Receptor Status, Status_Code")
    print()
    print("  Output files:")
    print(f"    {CLEAN_PATH}")
    print(f"    {DICT_PATH}")
    print(f"    {LOG_PATH}")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def run_pipeline():
    log.info("=" * 50)
    log.info("SG-CancerSight ETL Pipeline — START")
    log.info(f"Run timestamp: {datetime.now().isoformat()}")
    log.info("=" * 50)

    # 1. Load
    df_raw = load_raw(RAW_PATH)

    # 2. Validate
    val_report = validate(df_raw)

    # 3. Clean
    df_clean = clean(df_raw.copy(), val_report)

    # 4. Save clean data
    df_clean.to_csv(CLEAN_PATH, index=False)
    log.info(f"Clean data saved: {CLEAN_PATH}")

    # 5. Build and save data dictionary
    data_dict = build_data_dictionary(df_clean)
    with open(DICT_PATH, "w") as f:
        json.dump(data_dict, f, indent=2)
    log.info(f"Data dictionary saved: {DICT_PATH}")

    # 6. Print summary
    quality_summary(df_raw, df_clean, val_report)

    log.info("ETL Pipeline — COMPLETE")
    return df_clean, data_dict


if __name__ == "__main__":
    run_pipeline()
