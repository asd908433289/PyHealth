"""Local-only conversion of a SEER*Stat case listing into a PyHealth event table.

Do not commit the source, generated file, or PyHealth cache to GitHub.
The resulting CSV retains real patient IDs and is restricted research data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

IDENTIFIER = "Patient ID"
YEAR = "Year of diagnosis"
SITE = "Site recode ICD-O-3/WHO 2008"
MONTHS = "Survival months"
CAUSE = "SEER cause-specific death classification"

# Baseline categorical representation: not a full reproduction of the paper's
# continuous-variable normalization and feature selection. Clinical attributes
# must be verified as known at diagnosis before final experimental use.
FEATURE_COLUMNS = {
    "Age recode with <1 year olds and 90+": "age",
    "Sex": "sex",
    "Race recode (W, B, AI, API)": "race",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": "race_origin",
    "Histology recode - broad groupings": "histology",
    "Tumor Size Over Time Recode (1988+)": "tumor_size",
    "Summary stage 2000 (1998-2017)": "summary_stage",
    "Breast - Adjusted AJCC 6th Stage (1988-2015)": "breast_ajcc_stage",
    "Breast - Adjusted AJCC 6th T (1988-2015)": "breast_ajcc_t",
    "Breast - Adjusted AJCC 6th N (1988-2015)": "breast_ajcc_n",
    "Breast - Adjusted AJCC 6th M (1988-2015)": "breast_ajcc_m",
    "Derived AJCC Stage Group, 6th ed (2004-2015)": "derived_ajcc_stage",
    "Derived AJCC T, 6th ed (2004-2015)": "derived_ajcc_t",
    "Derived AJCC N, 6th ed (2004-2015)": "derived_ajcc_n",
    "Derived AJCC M, 6th ed (2004-2015)": "derived_ajcc_m",
    "Grade Recode (thru 2017)": "grade",
}


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    value = str(value).strip()
    if not value or value.casefold() in {"na", "n/a", "nan", "none"}:
        return "unknown"
    # The pipe is our token separator; never allow it inside a token.
    return value.replace("|", "/").replace("\n", " ")


def build_prepared_cases(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate cohort and create one event per cancer case.

    Args:
        raw: Labeled SEER*Stat CSV data within an authorized environment.

    Returns:
        DataFrame ready for PyHealth's yaml-defined event table.
    """
    required = {IDENTIFIER, YEAR, SITE, MONTHS, CAUSE, *FEATURE_COLUMNS}
    absent = sorted(required - set(raw.columns))
    if absent:
        raise ValueError(f"Required SEER columns missing: {absent}")

    yr = pd.to_numeric(raw[YEAR], errors="coerce")
    if not yr.between(2004, 2009).all():
        raise ValueError("Expected breast cancer diagnosis years 2004 through 2009")
    if not raw[SITE].astype(str).str.strip().eq("Breast").all():
        raise ValueError("Expected only Breast cases")
    if raw[IDENTIFIER].isna().any():
        raise ValueError("Missing patient IDs: cannot guarantee patient-level splitting")

    months = pd.to_numeric(raw[MONTHS], errors="coerce")
    # Original SEER code 9999 indicates unknown months; prevent labeling as 1.
    months = months.where(months.between(0, 9998))

    feature_columns = list(FEATURE_COLUMNS)
    prepared = pd.DataFrame({
        "patient_id": raw[IDENTIFIER].astype(str).str.strip(),
        "record_id": [f"case_{i:09d}" for i in range(len(raw))],
        # Year-only timestamp. PyHealth parses this via '%Y'; no diagnosis day is asserted.
        "diagnosis_year": yr.astype(int).astype(str),
        "survival_months": months,
        "cause_specific_death": raw[CAUSE].fillna("unknown").astype(str),
    })
    if prepared["patient_id"].eq("").any():
        raise ValueError("Blank patient identifiers")
    token_lists = []
    for values in raw[feature_columns].itertuples(index=False, name=None):
        tokens = [
            f"{FEATURE_COLUMNS[col]}={_clean_text(value)}"
            for col, value in zip(feature_columns, values)
        ]
        token_lists.append("|".join(tokens))
    prepared["feature_tokens"] = token_lists
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(
        args.input,
        low_memory=False,
        na_values=["NA", ""],
        dtype={IDENTIFIER: "string"},
    )
    prepared = build_prepared_cases(raw)
    prepared.to_csv(args.output, index=False)
    print(f"Records processed: {len(prepared):,}")
    print(f"Unique patients: {prepared['patient_id'].nunique():,}")
    print(f"Output columns: {list(prepared.columns)}")
    print("The generated file is still restricted SEER case-level data.")


if __name__ == "__main__":
    main()
