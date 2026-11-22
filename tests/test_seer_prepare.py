"""Synthetic-only checks for the SEER CSV conversion."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

script = Path(__file__).parents[1] / "examples" / "seer_prepare.py"
spec = importlib.util.spec_from_file_location("seer_prepare", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_preparation_with_synthetic_patient():
    row = {column: "Unknown" for column in module.FEATURE_COLUMNS}
    row.update({
        "Patient ID": "SYNTHETIC_1",
        "Year of diagnosis": "2006",
        "Site recode ICD-O-3/WHO 2008": "Breast",
        "Survival months": "72",
        "SEER cause-specific death classification": "Alive or dead of other cause",
    })
    prepared = module.build_prepared_cases(pd.DataFrame([row]))
    assert prepared.loc[0, "patient_id"] == "SYNTHETIC_1"
    assert prepared.loc[0, "survival_months"] == 72
    assert "grade=Unknown" in prepared.loc[0, "feature_tokens"]
    assert "survival_months=" not in prepared.loc[0, "feature_tokens"]


def test_rejects_wrong_cancer_type():
    row = {column: "Unknown" for column in module.FEATURE_COLUMNS}
    row.update({
        "Patient ID": "SYNTHETIC_1",
        "Year of diagnosis": "2006",
        "Site recode ICD-O-3/WHO 2008": "Lung and Bronchus",
        "Survival months": "30",
        "SEER cause-specific death classification": "0",
    })
    try:
        module.build_prepared_cases(pd.DataFrame([row]))
    except ValueError as err:
        assert "Breast" in str(err)
    else:
        raise AssertionError("Incorrect site should be rejected")
