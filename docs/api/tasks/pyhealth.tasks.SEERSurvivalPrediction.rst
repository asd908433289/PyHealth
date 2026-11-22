pyhealth.tasks.SEERSurvivalPrediction
=====================================

Overview
--------

Cancer-specific survival **binary classification**, at a 12-month or
60-month horizon, using events from
:class:`~pyhealth.datasets.seer.SEERDataset`. This is **not** a time-to-event
regression task and does not estimate a survival curve.

For each eligible case:

- ``1`` means observed survival time is at least the target horizon;
- ``0`` means cancer-attributable death occurred before the target horizon;
- cases without a determinable label (for example, right-censored or
  other-cause death before the horizon) are omitted.

The label logic follows Algorithm 1 of Hegselmann et al. (2018). The input
``features`` are categorical tokens encoded with a multi-hot processor.
Previously unseen validation/test tokens are ignored, so the model vocabulary
is learned from training patients only when split-aware processing is used.

The task's ``patient_id`` and ``record_id`` are metadata, never inputs to the
prediction model. Split by patient using ``PatientSplit`` to avoid leakage
between diagnosis records from the same person.

Examples
--------

.. code-block:: python

    from pyhealth.datasets import PatientSplit
    from pyhealth.datasets.seer import SEERDataset
    from pyhealth.tasks.seer_survival import SEERSurvivalPrediction

    dataset = SEERDataset(root="<approved_root>", cache_dir="<approved_cache>")
    train, val, test = dataset.set_task(
        SEERSurvivalPrediction(months=12),
        split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=42),
        num_workers=1,
    )

``months=60`` creates the five-year variant. Use only a data environment
explicitly permitted by the applicable SEER agreement.

API Reference
-------------

.. autoclass:: pyhealth.tasks.seer_survival.SEERSurvivalPrediction
   :members:
   :show-inheritance:

.. autofunction:: pyhealth.tasks.seer_survival.survival_label

.. autoclass:: pyhealth.tasks.seer_survival.SEEROOVMultiHotProcessor
   :members:
   :show-inheritance:
