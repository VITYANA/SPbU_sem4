import numpy as np
import pytest
from sklearn.datasets import make_classification  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.neighbors import KNeighborsClassifier  # type: ignore

from src.KNN.KNNClassifier import KNNClassifier


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=333,
    )
    return X, y


@pytest.mark.parametrize(
    "n_neighbors, leaf_size",
    [
        (1, 10),
        (3, 20),
        (5, 30),
    ],
)
def test_knn_classifier_predictions(n_neighbors, leaf_size, sample_data):
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn = KNNClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    knn.fit(X_train, y_train)

    sklearn_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    sklearn_knn.fit(X_train, y_train)

    our_predictions = knn.predict(X_test)

    sklearn_predictions = sklearn_knn.predict(X_test)

    assert np.array_equal(
        our_predictions, sklearn_predictions
    ), f"Predictions do not match for n_neighbors={n_neighbors}, leaf_size={leaf_size}"


def test_knn_classifier_more_neighbors_than_samples(sample_data):
    X, y = sample_data
    n_neighbors = len(X) + 1
    leaf_size = 10

    knn = KNNClassifier(n_neighbors=n_neighbors, leaf_size=leaf_size)
    knn.fit(X, y)

    predictions = knn.predict(X)

    most_common_class = np.bincount(y).argmax()
    assert np.all(
        predictions == most_common_class
    ), "Predictions should match the most common class when n_neighbors > n_samples"


@pytest.mark.parametrize("n_features", [2, 5, 10])
def test_knn_classifier_multidimensional_data(n_features):
    X, y = make_classification(
        n_samples=100,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    knn = KNNClassifier(n_neighbors=3, leaf_size=10)
    knn.fit(X_train, y_train)

    predictions = knn.predict(X_test)

    assert (
        predictions.shape == y_test.shape
    ), f"Predictions shape mismatch for n_features={n_features}"


def test_knn_classifier_empty_data():
    X = np.array([])
    y = np.array([])

    knn = KNNClassifier(n_neighbors=3, leaf_size=10)

    with pytest.raises(ValueError, match="List of points cannot be empty"):
        knn.fit(X, y)


@pytest.mark.parametrize("leaf_size", [0, -1])
def test_knn_classifier_invalid_leaf_size(leaf_size):
    X, y = make_classification(
        n_samples=10,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42,
    )

    knn = KNNClassifier(n_neighbors=3, leaf_size=leaf_size)

    with pytest.raises(AttributeError, match="Leaf size must be strictly positive"):
        knn.fit(X, y)
