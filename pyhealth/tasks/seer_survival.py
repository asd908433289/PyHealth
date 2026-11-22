"""Binary cancer-specific survival prediction tasks for SEER case listings.

This follows Algorithm 1 in Hegselmann et al. (MLHC 2018).
"""
from __future__ import annotations

from math import isfinite
from typing import Any

from pyhealth.tasks import BaseTask
from pyhealth.processors import MultiHotProcessor
import torch

# Confirm these labels using SEER's export dictionary when using a new release.
CANCER_DEATH_LABELS = {
    "1",
    "Dead (attributable to this cancer dx)",
}


def survival_label(
    survival_months: Any,
    cause_specific_death: Any,
    months: int,
) -> int | None:
    """Create a binary cancer-specific survival label (Algorithm 1).

    Args:
        survival_months: Observed follow-up duration in months. Missing
            or unknown duration codes cannot produce a label.
        cause_specific_death: SEER cause-specific classification category.
            Verify the accepted codes against the release's dictionary.
        months: Prediction horizon, either 12 or 60.

    Returns:
        ``1`` if survival duration reaches the prediction horizon;
        ``0`` for cancer-attributable death before that horizon; otherwise
        ``None`` for an indeterminate outcome.

    Raises:
        ValueError: If ``months`` is not supported.

    Note:
        These outcome fields are used to generate labels only, and must
        never be fed to a model as predictor variables.
    """
    if months not in (12, 60):
        raise ValueError("This reproduction supports only 12 and 60 months")
    try:
        duration = float(survival_months)
    except (ValueError, TypeError):
        return None
    if not isfinite(duration) or not (0 <= duration <= 9998):
        return None
    if duration >= months:
        return 1
    if str(cause_specific_death).strip() in CANCER_DEATH_LABELS:
        return 0
    return None


class SEEROOVMultiHotProcessor(MultiHotProcessor):
    """Multi-hot encode with a training-only feature vocabulary.

    Unseen categorical tokens in validation/test data are skipped rather
    than causing a ``KeyError`` or leaking held-out information into the
    vocabulary. This may discard information from rare categories.
    """

    def process(self, value: Any) -> torch.Tensor:
        """Return a float32 multi-hot vector for known feature tokens.

        Args:
            value: List of categorical feature tokens for one case.

        Returns:
            Tensor with one entry per token in the fitted vocabulary.

        Raises:
            ValueError: If the input is not a list.
        """
        if not isinstance(value, list):
            raise ValueError("Expected a list of categorical feature tokens")
        encoded = torch.zeros(len(self.label_vocab), dtype=torch.float32)
        for token in value:
            index = self.label_vocab.get(token)
            if index is not None:
                encoded[index] = 1.0
        return encoded


class SEERSurvivalPrediction(BaseTask):
    """Build one 12-/60-month prediction sample per eligible SEER case.

    The output target is ``1`` for observed survival through the horizon,
    ``0`` for earlier cancer-attributable death, and otherwise omitted.
    Model inputs consist only of categorical feature tokens.

    Args:
        months: Prediction horizon, 12 or 60 months.

    Note:
        Use ``PatientSplit`` to keep the same patient's case records
        together when splitting data and fitting feature processors.
    """

    input_schema = {"features": SEEROOVMultiHotProcessor}
    output_schema = {"label": "binary"}

    def __init__(self, months: int = 12) -> None:
        """Initialize the cancer-survival prediction task.

        Args:
            months: Survival horizon of 12 or 60 months.

        Raises:
            ValueError: If the horizon is unsupported.
        """
        super().__init__()
        if months not in (12, 60):
            raise ValueError("Prediction horizon must be 12 or 60 months")
        self.months = months
        self.task_name = f"SEERSurvival_{months}Months"

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Turn a patient's cancer events into labeled model samples.

        Args:
            patient: PyHealth patient with events in ``seer_cases``.

        Returns:
            List of eligible samples; empty if none can be labeled.
        """
        samples: list[dict[str, Any]] = []
        for event in patient.get_events("seer_cases"):
            label = survival_label(
                event.survival_months,
                event.cause_specific_death,
                self.months,
            )
            if label is None:
                continue
            tokens = str(event.feature_tokens).split("|")
            samples.append({
                "patient_id": str(patient.patient_id),
                "record_id": str(event.record_id),
                "visit_id": str(event.record_id),
                "features": tokens,
                "label": label,
            })
        return samples
