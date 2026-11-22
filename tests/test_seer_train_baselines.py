"""Synthetic-only unit tests for the SEER Logistic Regression baseline script."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import sparse
from sklearn.dummy import DummyClassifier

SOURCE = Path(__file__).resolve().parents[1] / "examples" / "seer_train_baselines.py"
if not SOURCE.is_file():
    SOURCE = Path(__file__).with_name("seer_train_baselines.py")
spec = importlib.util.spec_from_file_location("seer_train_baselines", SOURCE)
baseline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(baseline)


def test_stream_to_sparse_matrix_and_split_isolation():
    train_batches = [{
        "features": torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float),
        "label": torch.tensor([[1], [0]]),
        "patient_id": ["synthetic_A", "synthetic_A"],
    }]
    test_batches = [{
        "features": torch.tensor([[0, 0, 1]], dtype=torch.float),
        "label": torch.tensor([[0]]),
        "patient_id": ["synthetic_B"],
    }]
    train_X, train_y, train_ids = baseline.to_csr_and_labels(
        train_batches, split_name="train"
    )
    test_X, test_y, test_ids = baseline.to_csr_and_labels(
        test_batches, split_name="test"
    )
    assert train_X.shape == (2, 3)
    assert sparse.isspmatrix_csr(train_X)
    assert train_y.tolist() == [1, 0]
    assert test_y.tolist() == [0]
    baseline.assert_isolated_splits(train_ids, test_ids)
    with pytest.raises(RuntimeError, match="overlap"):
        baseline.assert_isolated_splits(train_ids, {"synthetic_A"})


def test_majority_class_metrics_with_cancer_deaths():
    X = sparse.csr_matrix(np.array([[0], [1], [0], [1], [1]]))
    y = np.array([1, 1, 1, 0, 0])
    dummy = DummyClassifier(strategy="most_frequent").fit(X, y)
    scores = baseline.evaluate_binary(
        y, dummy.predict(X), baseline.class_probability(dummy, X, 1)
    )
    assert scores["accuracy"] == pytest.approx(0.6)
    assert scores["roc_auc"] == pytest.approx(0.5)
    assert scores["recall_death"] == 0
    assert scores["f1_death"] == 0
    assert scores["true_death_pred_survival"] == 2
    assert scores["true_survival_pred_survival"] == 3


def test_logistic_tuning_uses_validation_not_test():
    train_X = sparse.csr_matrix(np.array([
        [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]
    ], dtype=np.float32))
    train_y = np.array([0, 1, 0, 1, 0, 1])
    model, c = baseline.select_logistic(
        train_X, train_y, train_X, train_y, [0.1, 1.0], max_iter=100
    )
    assert c in [0.1, 1.0]
    assert model.predict(train_X).shape == (6,)
    assert np.isfinite(baseline.class_probability(model, train_X, 1)).all()


def test_single_class_validation_returns_nan_auc():
    y = np.array([1, 1])
    scores = baseline.evaluate_binary(y, y, np.array([0.9, 0.9]))
    assert np.isnan(scores["roc_auc"])
    assert scores["f1_survival"] == 1


def test_invalid_c_values_rejected():
    with pytest.raises(Exception):
        baseline.parse_cs("0,-1")
    assert baseline.parse_cs("0.1,1,10") == [0.1, 1.0, 10.0]
