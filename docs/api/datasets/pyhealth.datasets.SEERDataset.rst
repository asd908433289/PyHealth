pyhealth.datasets.SEERDataset
=============================

Overview
--------

``SEERDataset`` loads a **locally prepared, access-controlled** case-listing
export from the US National Cancer Institute's Surveillance, Epidemiology,
and End Results (SEER) program. It uses PyHealth's ``BaseDataset`` and a
YAML-described ``seer_cases`` table. The initial implementation covers breast
cancer diagnoses from 2004--2009 and the November 2025 SEER Research Data
submission, not the 2016 data release used by the source publication.

A corresponding :class:`~pyhealth.tasks.seer_survival.SEERSurvivalPrediction`
task creates binary cancer-specific survival targets for 12 or 60 months.

Getting the data
----------------

- **Source:** `NCI SEER Research Data <https://seer.cancer.gov/data/access.html>`_.
- **Access:** registration and agreement to the applicable SEER Data Use
  Agreement; access to the original 2016 release is not supplied by this
  adapter.
- **Privacy:** this repository does not include patient-level SEER data,
  generated tensors, caches or trained weights. Use only an approved local
  or institutional research environment for patient-level experiments.
- **Preparation:** create a SEER*Stat case-listing selection for malignant
  breast cancers diagnosed in 2004--2009 using the appropriate cause-specific
  survival exclusions. Export **variable names and case values** as a
  delimited text file. Review the SEER*Stat dictionary when mapping codes.
- **Prepared layout:** run ``examples/seer_prepare.py`` locally to generate:

  .. code-block:: text

      <approved_root>/
          seer_cases.csv

  The prepared CSV contains ``patient_id``, ``record_id``, ``diagnosis_year``,
  ``survival_months``, ``cause_specific_death`` and ``feature_tokens``. The
  first two fields are identifiers and **must never become model features**.

See :doc:`../../seer_survival` for exact PowerShell commands, cohort counts,
model evaluation and limitations.

API Reference
-------------

.. autoclass:: pyhealth.datasets.seer.SEERDataset
   :members:
   :show-inheritance:
