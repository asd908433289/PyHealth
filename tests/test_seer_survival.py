"""Synthetic-only checks for the SEER Task (requires local PyHealth installation)."""
from __future__ import annotations

import pytest

pytest.importorskip("pyhealth")
from pyhealth.tasks.seer_survival import SEERSurvivalPrediction, survival_label


@pytest.mark.parametrize(
    "duration,cause,months,expected",
    [
        (11, "Dead (attributable to this cancer dx)", 12, 0),
        (12, "0", 12, 1),
        (11, "Dead (attributable to this cancer dx)", 60, 0),
        (24, "Dead (attributable to this cancer dx)", 60, 0),
        (60, "0", 60, 1),
        (11, "Dead (not attributable to this cancer dx)", 12, None),
        (11, "unknown", 12, None),
        (9999, "0", 12, None),
        (None, "1", 12, None),
    ],
)
def test_label(duration, cause, months, expected):
    assert survival_label(duration, cause, months) == expected


class FakeEvent:
    survival_months = "18"
    cause_specific_death = "Dead (not attributable to this cancer dx)"
    record_id = "synthetic_case_1"
    feature_tokens = "sex=Female|summary_stage=Localized"


class FakePatient:
    patient_id = "synthetic_patient"

    def get_events(self, name):
        assert name == "seer_cases"
        return [FakeEvent()]


def test_task_filters_censored_sample():
    assert SEERSurvivalPrediction(60)(FakePatient()) == []


def test_task_produces_sample():
    samples = SEERSurvivalPrediction(12)(FakePatient())
    assert len(samples) == 1
    assert samples[0]["label"] == 1
    assert samples[0]["features"] == ["sex=Female", "summary_stage=Localized"]
