import numpy as np
import pytest

from src.KNN.Scalers import MinMaxScaler


class TestMinMaxScaler:
    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            (
                np.array([[1, 2], [3, 4], [5, 6]]),
                np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            ),
            (
                np.array([[0, 0], [10, 10], [20, 20]]),
                np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            ),
            (
                np.array([[-1, -1], [0, 0], [1, 1]]),
                np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]),
            ),
            (
                np.array([[1, 5], [2, 4], [3, 3]]),
                np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]),
            ),
        ],
    )
    def test_min_max_scaler_fit_transform(self, data, expected_scaled_data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)

    @pytest.mark.parametrize(
        "data, expected_scaled_data",
        [
            (np.array([[5, 5]]), np.array([[0.0, 0.0]])),
            (
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            ),
            (
                np.array([[10, 10], [10, 10], [10, 10]]),
                np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            ),
        ],
    )
    def test_min_max_scaler_boundary(self, data, expected_scaled_data):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)

    def test_min_max_scaler_fit_transform_separate(self):
        data = np.array([[1, 2], [3, 4], [5, 6]])
        expected_scaled_data = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        scaler = MinMaxScaler()
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        np.testing.assert_array_almost_equal(scaled_data, expected_scaled_data)

    def test_min_max_scaler_transform_before_fit_error(self):
        scaler = MinMaxScaler()
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            scaler.transform(data)
