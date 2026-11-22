SEER Cancer Survival Prediction: Adapted Reproduction
=====================================================

Purpose
-------

This example reproduces a **subset of the methodology** in Hegselmann,
Greulich, Varghese and Dugas (2018), *Reproducible Survival Prediction with
SEER Cancer Data*, using PyHealth 2.x. It builds an input pipeline for
breast cancer cases diagnosed in 2004--2009, constructs 12- and 60-month
cancer-specific survival labels, and compares a majority-class baseline,
Logistic Regression and a multilayer perceptron (MLP).

This is an **adapted reproduction**, not an exact numerical reproduction of
the publication. The original experiment used the November 2016 SEER
submission, a substantially larger original feature inventory, numerical
normalization plus one-hot encodings, and also studied lung cancer. This
implementation uses November 2025 SEER Research Data and 16 selected
categorical feature groups. Do not interpret score differences as evidence
of an identical cohort or identical preprocessing.

Access and privacy
------------------

SEER is a public-interest registry available to eligible researchers after
registering and accepting the applicable Data Use Agreement; the underlying
individual case records are **not open redistribution data**. Confirm that
your machine-learning use and computing environment are authorized.

- Never commit original case-listing exports (``seer_breast_*.txt``),
  ``seer_cases.csv``, patient IDs, task caches, patient-level predictions or
  fitted model weights.
- Store case-level data and the PyHealth cache outside the Git checkout,
  within approved research storage.
- Documentation, source code, aggregate metrics and **synthetic** test
  fixtures can be shared in a PR subject to the applicable agreement.

Data and preprocessing
----------------------

Cohort: malignant breast cancer cases diagnosed between 2004 and
2009. The SEER*Stat survival session should select known age and exclude
death-certificate-only/autopsy-only cases, alive cases with no survival time,
unknown/missing cause of death and applicable expected-survival-table cases.
The ``Multiple Primary Selection`` rule can change which tumor records
contribute and must be recorded. The included dataset's selection parameters
are not equivalent to the historical 2016 cohort; a published original
SEER*Stat session remains the source of truth for an exact historical run.

``examples/seer_prepare.py`` expects a **header row** and the exact
November 2025 export variable names listed in its ``FEATURE_COLUMNS``. It
validates years and breast site, retains the patient identifier strictly for
patient-level splitting, and converts 16 baseline clinical fields to
``feature=value`` categorical tokens separated by ``|``.

The prepared table has these columns:

.. list-table::
   :header-rows: 1

   * - Field
     - Role
   * - ``patient_id``
     - Splitting and bookkeeping (not a predictor)
   * - ``record_id``
     - Case-level bookkeeping (not a predictor)
   * - ``diagnosis_year``
     - Year-only PyHealth event timestamp
   * - ``survival_months``
     - Label generation only; not a predictor
   * - ``cause_specific_death``
     - Label generation only; not a predictor
   * - ``feature_tokens``
     - Multi-hot baseline predictors (16 categorical field groups)

The included baseline has a loss of detail compared with the paper:
continuous variables (for example, tumor size) are treated as categorical
codes, some coarse recodes replace more detailed attributes, and each
unseen test-set token is ignored rather than represented with a learned
``unknown`` category. Staging features are potentially redundant and must be
interpreted with caution. The pipeline intentionally excludes follow-up
status and other known outcome fields from the predictors to prevent leakage.

Label rules
-----------

For a horizon ``n`` of 12 or 60 months, the task applies:

.. code-block:: text

    if survival_months < n and death_is_due_to_this_cancer:
        label = 0
    elif survival_months >= n:
        label = 1
    else:
        omit the case

SEER's exported strings must be verified against the release-specific data
dictionary; ``survival_label`` accepts ``1`` and
``Dead (attributable to this cancer dx)`` as cancer-death codes. In the
original paper, ambiguous labels were excluded. The 60-month and 12-month
cohorts therefore have different denominators.

Installation and commands
-------------------------

Run from the root of the **PyHealth checkout** with Python 3.12 or 3.13 and
its declared dependencies installed. The ``<approved_root>`` and
``<approved_cache>`` values below stand for approved paths *outside* the
source tree. In Windows PowerShell:

.. code-block:: powershell

    python -m pip install -e .
    python -m pytest tests/test_seer_prepare.py tests/test_seer_survival.py tests/test_seer_oov_processor.py tests/test_seer_train_baselines.py -q
    python examples/seer_prepare.py --input "<approved_root>/seer_breast_2004_2009.txt" --output "<approved_root>/seer_cases.csv"
    python examples/seer_survival_smoke.py --root "<approved_root>" --cache "<approved_cache>" --months 12 --dev
    python examples/seer_survival_smoke.py --root "<approved_root>" --cache "<approved_cache>" --months 60 --dev

To run on the complete selected cohort, omit ``--dev``. Training commands:

.. code-block:: powershell

    python examples/seer_train_mlp.py --root "<approved_root>" --cache "<approved_cache>" --months 12 --epochs 3
    python examples/seer_train_mlp.py --root "<approved_root>" --cache "<approved_cache>" --months 60 --epochs 3
    python examples/seer_train_baselines.py --root "<approved_root>" --cache "<approved_cache>" --months 12 --c-values "0.1,1,10"
    python examples/seer_train_baselines.py --root "<approved_root>" --cache "<approved_cache>" --months 60 --c-values "0.1,1,10"

For the **class-weighting ablation**, change the class weights while keeping
the same horizon, random seed and patient split. Run the regular and weighted
versions together using the rubric-named wrapper:

.. code-block:: powershell

    python examples/seer_survival_prediction_logistic.py --root "<approved_root>" --cache "<approved_cache>" --months 12 --ablation
    python examples/seer_survival_prediction_logistic.py --root "<approved_root>" --cache "<approved_cache>" --months 60 --ablation

Model and evaluation methodology
--------------------------------

- Split by **patient**, not by individual cancer case, into training,
  validation and test (80/10/10; random seed 42).
- Fit the multi-hot vocabulary on training patients only, and ignore tokens
  that appear only in validation or test data.
- Tune Logistic Regression regularization ``C`` using the validation
  ROC-AUC plus **survivor-class F1** objective when both classes are present.
  The script uses validation log loss if the validation subset contains just
  one class.
- Train the MLP for a small number of epochs and select its checkpoint using
  validation performance, not test labels.
- Report ROC-AUC, survivor F1 and accuracy, plus survivor/death class
  precision-recall metrics. **A high survivor F1 can hide poor cancer-death
  recall** when the positive survivor class is very common.
- The optional class-weighted Logistic Regression ablation uses
  ``class_weight="balanced"`` and retains the same split as ordinary LR.

Observed local breast-cancer results
------------------------------------

The numbers below are **aggregates reported from the course implementation's
local November 2025 SEER runs**. They are not calculated by the docs build,
not clinical validation, and not the original publication's results. Full
MLP and LR comparisons should still verify identical patient splits and input
representation. The 12-month valid cohort had 310,856 samples; the 60-month
cohort had 289,740.

.. list-table:: Held-out ROC-AUC (adapted breast-only reproduction)
   :header-rows: 1

   * - Model
     - 12 months
     - 60 months
   * - MLP
     - 0.9254
     - 0.8769
   * - Logistic Regression
     - 0.9258
     - 0.8784
   * - Logistic Regression, class-weighted
     - 0.9249
     - 0.8783

.. list-table:: Cancer-death classification (Logistic Regression)
   :header-rows: 1

   * - Horizon
     - Class weights
     - Death recall
     - Death precision
     - Death F1
   * - 12 months
     - None
     - 0.1795
     - 0.5937
     - 0.2756
   * - 12 months
     - Balanced
     - 0.8551
     - 0.1919
     - 0.3134
   * - 60 months
     - None
     - 0.4196
     - 0.7306
     - 0.5330
   * - 60 months
     - Balanced
     - 0.7712
     - 0.3919
     - 0.5197

Class weighting improved cancer-death recall but reduced precision and
accuracy. It is a **novel implemented ablation for this project**, not an
ablation reported by the 2018 paper. The choice of weighting depends on how
false negatives and false positives are valued; these experiments cannot
establish clinical suitability.

Tests and reproducibility
-------------------------

The synthetic tests cover the data-preparation schema, survival label
boundaries and exclusions, unseen-category processor behavior, binary
metrics and baseline helpers. They do not require or distribute real SEER
records. The local execution evidence supplied for this project reported
**21 passing tests**; this number should be refreshed after subsequent code
changes. A full smoke test must additionally run against the authorized
prepared cohort, because synthetic unit tests alone do not verify the entire
PyHealth environment.

Further limitations include year-only timestamps (not a claimed exact
calendar date), local SEER export label vocabularies, registry/release
differences, an original numerical feature set that has not been fully
recreated, breast cancer only (no lung cohort) and no clinical validation.

References
----------

- Hegselmann, S.; Greulich, L.; Varghese, J.; and Dugas, M. 2018.
  `Reproducible Survival Prediction with SEER Cancer Data
  <https://proceedings.mlr.press/v85/hegselmann18a.html>`_.
- `Original authors' code and SEER*Stat session files
  <https://github.com/stefanhgm/MLHC2018-reproducible-survival-seer>`_.
- `SEER Research Data access <https://seer.cancer.gov/data/access.html>`_.
