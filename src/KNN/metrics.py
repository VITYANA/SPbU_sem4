import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))

    precision = true_positives / (true_positives + false_positives + 1e-10)
    recall = true_positives / (true_positives + false_negatives + 1e-10)

    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    return f1
