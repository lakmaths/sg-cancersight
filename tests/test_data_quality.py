"""
SG-CancerSight — Phase 1: Data Quality Tests
=============================================
Automated pytest tests for the cleaned SEER dataset.
Run: pytest tests/test_data_quality.py -v

These tests simulate the validation checks that would be
required before any clinical/epidemiological analysis at NCCS.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

CLEAN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "seer_clean.csv"


@pytest.fixture(scope="module")
def df():
    """Load the cleaned dataset once for all tests."""
    assert CLEAN_PATH.exists(), f"Clean data not found: {CLEAN_PATH}. Run etl_pipeline.py first."
    return pd.read_csv(CLEAN_PATH)


# ── Structural tests ───────────────────────────────────────────────────────

def test_dataset_not_empty(df):
    """Dataset must have rows."""
    assert len(df) > 0, "Dataset is empty"

def test_minimum_row_count(df):
    """Should retain at least 95% of raw records after cleaning."""
    assert len(df) >= 3800, f"Unexpectedly few rows: {len(df)}"

def test_required_columns_present(df):
    """All analysis-critical columns must exist."""
    required = [
        "Age", "Race", "T Stage", "N Stage", "6th Stage", "Grade",
        "Tumor Size", "Regional Node Examined", "Regional Node Positive",
        "Estrogen Status", "Progesterone Status",
        "Survival Months", "Status", "Status_Code",
        "Age Group", "Stage Group", "Receptor Status"
    ]
    missing_cols = [c for c in required if c not in df.columns]
    assert not missing_cols, f"Missing required columns: {missing_cols}"


# ── Age tests ──────────────────────────────────────────────────────────────

def test_age_range_valid(df):
    """All ages must be between 18 and 100."""
    assert df["Age"].between(18, 100).all(), \
        f"Ages out of range: {df[~df['Age'].between(18,100)]['Age'].tolist()}"

def test_age_group_no_null(df):
    """Derived Age Group must have no nulls."""
    assert df["Age Group"].isna().sum() == 0, "Age Group has missing values"

def test_age_group_valid_labels(df):
    """Age Group labels must be one of the expected bands."""
    valid = {"<40", "40–49", "50–59", "60–69", "70+"}
    found = set(df["Age Group"].astype(str).unique())
    unexpected = found - valid
    assert not unexpected, f"Unexpected Age Group values: {unexpected}"


# ── Survival tests ─────────────────────────────────────────────────────────

def test_survival_months_positive(df):
    """All survival months must be >= 1."""
    assert (df["Survival Months"] >= 1).all(), \
        f"Negative/zero survival months found: {df[df['Survival Months']<1]['Survival Months'].tolist()}"

def test_survival_months_not_null(df):
    """Survival months cannot be null — it is the primary outcome."""
    assert df["Survival Months"].isna().sum() == 0

def test_status_code_binary(df):
    """Status_Code must be 0 or 1 only."""
    assert set(df["Status_Code"].unique()).issubset({0, 1}), \
        f"Status_Code has values other than 0/1: {df['Status_Code'].unique()}"

def test_status_consistent_with_code(df):
    """Status='Dead' must map to Status_Code=1 and vice versa."""
    dead_mask = df["Status"] == "Dead"
    assert (df.loc[dead_mask, "Status_Code"] == 1).all(), "Dead patients coded as 0"
    assert (df.loc[~dead_mask, "Status_Code"] == 0).all(), "Alive patients coded as 1"


# ── Tumour / clinical tests ────────────────────────────────────────────────

def test_tumor_size_no_sentinel(df):
    """Sentinel value 999 must have been recoded to NaN."""
    assert (df["Tumor Size"] != 999).all(), "Sentinel value 999 still present in Tumor Size"

def test_tumor_size_positive(df):
    """All non-missing tumour sizes must be positive."""
    non_null = df["Tumor Size"].dropna()
    assert (non_null > 0).all(), "Non-positive tumour size found"

def test_nodes_positive_le_examined(df):
    """Nodes positive cannot exceed nodes examined."""
    both_present = df[df["Regional Node Examined"].notna() & df["Regional Node Positive"].notna()]
    violation = both_present[both_present["Regional Node Positive"] > both_present["Regional Node Examined"]]
    assert len(violation) == 0, f"{len(violation)} rows where nodes positive > examined"


# ── Categorical integrity tests ────────────────────────────────────────────

def test_stage_values_valid(df):
    """6th Stage must only contain expected AJCC values."""
    valid = {"I", "IIA", "IIB", "IIIA", "IIIB", "IIIC"}
    unexpected = set(df["6th Stage"].unique()) - valid
    assert not unexpected, f"Unexpected 6th Stage values: {unexpected}"

def test_stage_group_values_valid(df):
    """Derived Stage Group must be Stage I, II, or III."""
    valid = {"Stage I", "Stage II", "Stage III"}
    unexpected = set(df["Stage Group"].unique()) - valid
    assert not unexpected, f"Unexpected Stage Group values: {unexpected}"

def test_t_stage_valid(df):
    valid = {"T1", "T2", "T3", "T4"}
    unexpected = set(df["T Stage"].unique()) - valid
    assert not unexpected, f"Unexpected T Stage values: {unexpected}"

def test_n_stage_valid(df):
    valid = {"N0", "N1", "N2", "N3"}
    unexpected = set(df["N Stage"].unique()) - valid
    assert not unexpected, f"Unexpected N Stage values: {unexpected}"

def test_er_status_valid(df):
    valid = {"Positive", "Negative"}
    unexpected = set(df["Estrogen Status"].unique()) - valid
    assert not unexpected, f"Unexpected ER Status: {unexpected}"

def test_pr_status_valid(df):
    valid = {"Positive", "Negative"}
    unexpected = set(df["Progesterone Status"].unique()) - valid
    assert not unexpected, f"Unexpected PR Status: {unexpected}"

def test_receptor_status_valid(df):
    """Derived receptor status must have two categories."""
    valid = {"Hormone Receptor+", "Triple Negative / HR-"}
    unexpected = set(df["Receptor Status"].unique()) - valid
    assert not unexpected, f"Unexpected Receptor Status: {unexpected}"


# ── Missing value tests ────────────────────────────────────────────────────

def test_key_clinical_columns_no_missing(df):
    """Core clinical columns used in survival analysis must be complete."""
    critical_cols = ["Age", "6th Stage", "Estrogen Status", "Status", "Status_Code", "Survival Months"]
    for col in critical_cols:
        missing_n = df[col].isna().sum()
        assert missing_n == 0, f"Critical column '{col}' has {missing_n} missing values"

def test_missing_within_acceptable_threshold(df):
    """No column should have more than 10% missing values post-cleaning."""
    for col in df.columns:
        pct_missing = df[col].isna().sum() / len(df) * 100
        assert pct_missing <= 10.0, \
            f"Column '{col}' has {pct_missing:.1f}% missing — exceeds 10% threshold"


# ── Distribution sanity tests ──────────────────────────────────────────────

def test_stage_distribution_reasonable(df):
    """Stage I should be the most common — early detection in registry."""
    stage_counts = df["Stage Group"].value_counts(normalize=True)
    assert stage_counts.get("Stage I", 0) > 0.05, "Unexpectedly few Stage I cases"

def test_er_positive_majority(df):
    """ER+ should be >60% in breast cancer registries."""
    er_pos_pct = (df["Estrogen Status"] == "Positive").mean() * 100
    assert er_pos_pct > 60, f"ER+ rate unexpectedly low: {er_pos_pct:.1f}%"

def test_death_rate_plausible(df):
    """Death rate should be between 10% and 70% in a registry cohort."""
    death_rate = df["Status_Code"].mean() * 100
    assert 10 <= death_rate <= 70, f"Death rate implausible: {death_rate:.1f}%"
