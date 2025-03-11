import numpy as np
import pytest

from src.KNN.split_train_data import train_test_split


class TestTrainTestSplit:
    @pytest.mark.parametrize(
        "X, y, test_size, random_state",
        [
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), 0.4, 0),
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), 0.33, 42),
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), 0.0, 42),
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), 1.0, 42),
        ],
    )
    def test_train_test_split_properties(self, X, y, test_size, random_state):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size, random_state
        )

        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)

        assert np.isclose(
            (-1 * 10 * len(X_test) // len(X) * -1 / 10), test_size, atol=0.01
        ) or np.isclose(len(X_test) / len(X), test_size, atol=0.01)

    @pytest.mark.parametrize(
        "X, y, test_size, n_train_X, n_test_X, n_train_y, n_test_y",
        [
            (
                np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]),
                np.array([0, 1, 0, 1, 0]),
                0.3,
                3,
                2,
                3,
                2,
            ),
        ],
    )
    def test_train_test_split_lengths(
        self, X, y, test_size, n_train_X, n_test_X, n_train_y, n_test_y
    ):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size)

        assert len(X_train) == n_train_X
        assert len(X_test) == n_test_X
        assert len(y_train) == n_train_y
        assert len(y_test) == n_test_y

    @pytest.mark.parametrize(
        "X, y, test_size, random_state",
        [
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), 2.0, 42),
            (np.array([[1, 2], [3, 4], [5, 6]]), np.array([0, 1, 0]), -0.1, 42),
        ],
    )
    def test_train_test_split_invalid_test_size(self, X, y, test_size, random_state):
        with pytest.raises(ValueError):
            train_test_split(X, y, test_size, random_state)

    def test_train_test_split_deterministic(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 1, 0, 1, 0])
        test_size = 0.3
        random_state = 42

        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            X, y, test_size, random_state
        )
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            X, y, test_size, random_state
        )

        np.testing.assert_array_equal(X_train1, X_train2)
        np.testing.assert_array_equal(X_test1, X_test2)
        np.testing.assert_array_equal(y_train1, y_train2)
        np.testing.assert_array_equal(y_test1, y_test2)
