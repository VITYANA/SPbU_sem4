import numpy as np
import pytest

from src.KNN.metrics import accuracy, f1_score


def test_accuracy_perfect_prediction():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_all_wrong():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    assert accuracy(y_true, y_pred) == 0.0


def test_accuracy_partially_correct():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    assert accuracy(y_true, y_pred) == 0.8


def test_accuracy_empty_input():
    y_true = np.array([])
    y_pred = np.array([])
    assert np.isnan(accuracy(y_true, y_pred))


def test_accuracy_different_lengths():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        accuracy(y_true, y_pred)


def test_f1_score_perfect_prediction():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    assert np.isclose(f1_score(y_true, y_pred), 1.0)


def test_f1_score_all_wrong():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0])
    assert f1_score(y_true, y_pred) == 0.0


def test_f1_score_partially_correct():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    assert np.isclose(f1_score(y_true, y_pred), 0.8)


def test_f1_score_no_positive_predictions():
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    assert f1_score(y_true, y_pred) == 0.0


def test_f1_score_empty_input():
    y_true = np.array([])
    y_pred = np.array([])
    assert f1_score(y_true, y_pred) == 0.0


def test_f1_score_different_lengths():
    y_true = np.array([1, 0, 1])
    y_pred = np.array([1, 0])
    with pytest.raises(ValueError):
        f1_score(y_true, y_pred)
