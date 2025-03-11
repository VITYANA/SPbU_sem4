from typing import Optional, Tuple

import numpy as np


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if test_size < 0 or test_size > 1:
        raise ValueError("test_size must be in [0, 1]")

    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    n_test = round(n_samples * test_size)
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
