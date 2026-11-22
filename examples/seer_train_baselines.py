"""SEER survival baselines: majority classifier and Logistic Regression.

Run from the PyHealth repository root, with the existing SEERDataset and
SEERSurvivalPrediction implementations installed. Only summary metrics are
printed. Patient-level data, cache files, and model weights are not exported.

This is an ADAPTED reproduction (SEER Nov 2025 release and categorical
multi-hot representation), not the original paper's 108-feature pipeline.
"""

from __future__ import annotations

import argparse
import random
import warnings
from typing import Any, Iterable

import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

SEED = 42


def class_probability(model: Any, X: sparse.csr_matrix, label: int) -> np.ndarray:
    """Return P(label) even if a dummy model learned only one class."""
    classes = list(model.classes_)
    if label not in classes:
        return np.zeros(X.shape[0], dtype=np.float64)
    return np.asarray(model.predict_proba(X)[:, classes.index(label)], dtype=np.float64)


def evaluate_binary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    p_survived: np.ndarray,
) -> dict[str, float | int]:
    """Score both survival=1 and cancer death=0 without exposing records."""
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=int).reshape(-1)
    p_survived = np.asarray(p_survived, dtype=float).reshape(-1)
    if not (len(y_true) == len(y_pred) == len(p_survived)):
        raise ValueError("Prediction, probability, and label lengths differ")
    if not len(y_true) or not np.isin(y_true, [0, 1]).all():
        raise ValueError("Expected non-empty binary survival labels (0 or 1)")
    if not np.isfinite(p_survived).all():
        raise ValueError("Predicted probabilities must be finite")
    p_survived = np.clip(p_survived, 0.0, 1.0)
    both_classes = np.unique(y_true).size == 2
    n_deaths = int(np.sum(y_true == 0))
    n_survived = int(np.sum(y_true == 1))
    # The confusion matrix ordering is explicit: 0=cancer death, 1=survived.
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    results: dict[str, float | int] = {
        "samples": len(y_true),
        "cancer_deaths": n_deaths,
        "survived": n_survived,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_survival": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_death": float(
            precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "recall_death": float(
            recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        ),
        "f1_death": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "true_death_pred_death": int(matrix[0, 0]),
        "true_death_pred_survival": int(matrix[0, 1]),
        "true_survival_pred_death": int(matrix[1, 0]),
        "true_survival_pred_survival": int(matrix[1, 1]),
        "roc_auc": float("nan"),
        "pr_auc_survival": float("nan"),
        "pr_auc_death": float("nan"),
    }
    if both_classes:
        results["roc_auc"] = float(roc_auc_score(y_true, p_survived))
        results["pr_auc_survival"] = float(
            average_precision_score(y_true, p_survived)
        )
        results["pr_auc_death"] = float(
            average_precision_score(1 - y_true, 1.0 - p_survived)
        )
    return results


def print_scores(title: str, scores: dict[str, float | int]) -> None:
    """Print aggregate metrics only; do not print patient IDs or predictions."""
    print(f"\n=== {title} ===")
    print(f"Samples: {int(scores['samples']):,}")
    print(
        f"Class counts: cancer deaths={int(scores['cancer_deaths']):,}, "
        f"survived={int(scores['survived']):,}"
    )
    for key in (
        "roc_auc", "pr_auc_survival", "pr_auc_death", "accuracy",
        "f1_survival", "precision_death", "recall_death", "f1_death",
    ):
        print(f"{key}: {scores[key]:.4f}")
    print("Confusion matrix (rows=actual; columns=predicted):")
    print("                     death(0)  survived(1)")
    print(
        f"  actual death(0)    {scores['true_death_pred_death']:8d} "
        f"{scores['true_death_pred_survival']:12d}"
    )
    print(
        f"  actual survived(1) {scores['true_survival_pred_death']:8d} "
        f"{scores['true_survival_pred_survival']:12d}"
    )


def to_csr_and_labels(
    batches: Iterable[dict[str, Any]], *, split_name: str
) -> tuple[sparse.csr_matrix, np.ndarray, set[str]]:
    """Stream PyHealth's encoded multi-hot features into memory-efficient CSR.

    No feature vocabulary is fitted here: PyHealth already fitted it on the
    training patients inside ``set_task(split=PatientSplit(...))``.
    """
    blocks: list[sparse.csr_matrix] = []
    targets: list[np.ndarray] = []
    patient_ids: set[str] = set()
    for batch in batches:
        features = batch["features"]
        labels = batch["label"]
        if hasattr(features, "detach"):
            features = features.detach().cpu().numpy()
        if hasattr(labels, "detach"):
            labels = labels.detach().cpu().numpy()
        feats = np.asarray(features, dtype=np.float32)
        labels_np = np.asarray(labels).reshape(-1)
        if feats.ndim != 2 or feats.shape[0] != labels_np.size:
            raise ValueError(
                f"{split_name}: expected [batch, features] and [batch] labels; "
                f"got {feats.shape} and {labels_np.shape}"
            )
        if not np.isfinite(feats).all():
            raise ValueError(f"{split_name}: non-finite feature values")
        if not np.isin(labels_np, [0, 1]).all():
            raise ValueError(f"{split_name}: non-binary labels")
        blocks.append(sparse.csr_matrix(feats))
        targets.append(labels_np.astype(np.int8))
        if "patient_id" not in batch:
            raise ValueError(
                "PyHealth batch lacks patient_id; cannot verify patient isolation"
            )
        batch_patient_ids = batch["patient_id"]
        if len(batch_patient_ids) != labels_np.size:
            raise ValueError(f"{split_name}: patient_id count mismatch")
        patient_ids.update(str(p) for p in batch_patient_ids)
    if not blocks:
        raise ValueError(f"{split_name}: no batches received")
    X = sparse.vstack(blocks, format="csr", dtype=np.float32)
    y = np.concatenate(targets)
    print(
        f"{split_name}: {X.shape[0]:,} samples, {X.shape[1]:,} input "
        f"dimensions, {len(patient_ids):,} unique patients; "
        f"death={int(np.sum(y == 0)):,}, survived={int(np.sum(y == 1)):,}"
    )
    return X, y, patient_ids


def assert_isolated_splits(*patient_groups: set[str]) -> None:
    """Verify train/val/test patient IDs do not overlap."""
    for i, first in enumerate(patient_groups):
        for second in patient_groups[i + 1:]:
            if not first.isdisjoint(second):
                raise RuntimeError("Patient IDs overlap across dataset splits")


def select_logistic(
    train_X: sparse.csr_matrix,
    train_y: np.ndarray,
    val_X: sparse.csr_matrix,
    val_y: np.ndarray,
    c_values: list[float],
    *,
    class_weight_balanced: bool = False,
    max_iter: int = 300,
) -> tuple[LogisticRegression, float]:
    """Choose Logistic Regression C using validation ROC-AUC + survival F1.

    This matches the original paper's model-selection objective when AUC is
    defined. If the validation cohort has one class, use validation log loss.
    No testing labels are consulted during model selection.
    """
    if np.unique(train_y).size != 2:
        raise ValueError(
            "Training split contains only one class; rerun without --dev "
            "or use a larger development cohort."
        )
    has_both_val = np.unique(val_y).size == 2
    if not has_both_val:
        print("WARNING: validation has one class; selecting C by validation log loss.")
    winner: LogisticRegression | None = None
    winner_score = -float("inf")
    winner_c = c_values[0]
    for c in c_values:
        estimator = LogisticRegression(
            C=c,
            solver="liblinear",
            max_iter=max_iter,
            class_weight="balanced" if class_weight_balanced else None,
            random_state=SEED,
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", ConvergenceWarning)
            estimator.fit(train_X, train_y)
        if any(issubclass(w.category, ConvergenceWarning) for w in observed):
            print(f"WARNING: C={c:g} did not converge; increase --max-iter.")
        proba_val = class_probability(estimator, val_X, 1)
        pred_val = estimator.predict(val_X)
        if has_both_val:
            auc = float(roc_auc_score(val_y, proba_val))
            f1 = float(f1_score(val_y, pred_val, pos_label=1, zero_division=0))
            criterion = auc + f1
            print(f"LogReg C={c:g}: val AUC={auc:.4f}, F1-survival={f1:.4f}, sum={criterion:.4f}")
        else:
            criterion = -float(
                log_loss(val_y, np.clip(proba_val, 1e-7, 1 - 1e-7), labels=[0, 1])
            )
            print(f"LogReg C={c:g}: val negative log loss={criterion:.4f}")
        if criterion > winner_score:
            winner, winner_score, winner_c = estimator, criterion, c
    assert winner is not None
    return winner, winner_c


def parse_cs(text: str) -> list[float]:
    values = [float(t.strip()) for t in text.split(",") if t.strip()]
    if not values or any(not np.isfinite(x) or x <= 0 for x in values):
        raise argparse.ArgumentTypeError("--c-values needs positive finite numbers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Approved folder containing seer_cases.csv")
    parser.add_argument("--cache", required=True, help="Approved PyHealth cache directory")
    parser.add_argument("--months", type=int, choices=[12, 60], default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--c-values", type=parse_cs, default=[0.1, 1.0, 10.0])
    parser.add_argument("--class-weight-balanced", action="store_true",
                        help="Optional novel class-imbalance ablation")
    parser.add_argument("--dev", action="store_true", help="Small integration test only")
    args = parser.parse_args()

    # Import only here so the metric and matrix helpers can be unit-tested
    # on synthetic arrays without installing the full PyHealth repository.
    from pyhealth.datasets import PatientSplit, get_dataloader
    from pyhealth.datasets.seer import SEERDataset
    from pyhealth.tasks.seer_survival import SEERSurvivalPrediction

    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch
        torch.manual_seed(SEED)
    except ImportError:
        pass
    print(f"SEER breast cancer: {args.months}-month survival, seed={SEED}")
    dataset = SEERDataset(
        root=args.root, cache_dir=args.cache, num_workers=1, dev=args.dev
    )
    train_ds, val_ds, test_ds = dataset.set_task(
        SEERSurvivalPrediction(args.months),
        num_workers=1,
        split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=SEED),
    )
    print(
        "Train/validation/test samples: "
        f"{len(train_ds):,}/{len(val_ds):,}/{len(test_ds):,}"
    )
    inputs = []
    for name, part in (
        ("Train", train_ds), ("Validation", val_ds), ("Test", test_ds)
    ):
        loader = get_dataloader(part, batch_size=args.batch_size, shuffle=False)
        inputs.append(to_csr_and_labels(loader, split_name=name))
    (X_tr, y_tr, ids_tr), (X_va, y_va, ids_va), (X_te, y_te, ids_te) = inputs
    assert_isolated_splits(ids_tr, ids_va, ids_te)
    print("Patient overlap check: PASS (three sets are disjoint)")
    if X_tr.shape[1] != X_va.shape[1] or X_tr.shape[1] != X_te.shape[1]:
        raise ValueError("Feature vector dimensions differ between splits")

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_tr, y_tr)
    print_scores(
        "Held-out TEST: majority-class baseline",
        evaluate_binary(y_te, dummy.predict(X_te), class_probability(dummy, X_te, 1)),
    )

    model, chosen_c = select_logistic(
        X_tr, y_tr, X_va, y_va, args.c_values,
        class_weight_balanced=args.class_weight_balanced,
        max_iter=args.max_iter,
    )
    print(f"Chosen Logistic Regression C: {chosen_c:g}")
    print_scores(
        "Held-out TEST: Logistic Regression",
        evaluate_binary(y_te, model.predict(X_te), class_probability(model, X_te, 1)),
    )
    print("\nAdapted reproduction: Nov 2025 SEER, reduced categorical inputs.")
    print("No patient-level predictions or model weights were saved.")


if __name__ == "__main__":
    main()
