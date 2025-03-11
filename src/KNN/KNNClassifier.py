from collections import Counter
from typing import List

import numpy as np

from src.KNN.KDNode import Point
from src.KNN.KDTree import KDTree


class KNNClassifier:
    def __init__(self, n_neighbors: int, leaf_size: int = 30):
        self.n_neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.kdtree: KDTree | None = None
        self.labels: np.ndarray = np.array([])
        self.points: List[Point] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        self.points = [Point(coord) for coord in X]
        self.labels = y

        self.kdtree = KDTree(self.points, self.leaf_size)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.kdtree is None or self.labels is None:
            raise ValueError("Classifier is not fitted yet")

        test_points: List[Point] = [Point(coord) for coord in X]

        probabilities = []
        for point in test_points:
            neighbors = self.kdtree.query([point], self.n_neighbors)[point]

            neighbor_indices = [
                self._find_point_index(neighbor) for neighbor in neighbors
            ]

            neighbor_labels = self.labels[neighbor_indices]

            label_counts = Counter(neighbor_labels)
            total = sum(label_counts.values())
            prob = {label: count / total for label, count in label_counts.items()}
            probabilities.append(prob)

        unique_labels = np.unique(self.labels)
        prob_array = np.zeros((len(X), len(unique_labels)))
        for i, prob in enumerate(probabilities):
            for j, label in enumerate(unique_labels):
                if isinstance(label, (tuple, list, np.ndarray)):
                    label_key = str(label)
                else:
                    label_key = label

                prob_array[i, j] = prob.get(label_key, 0.0)

        return prob_array

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def _find_point_index(self, point: Point) -> int:
        for i, p in enumerate(self.points):
            if np.array_equal(p.coord, point.coord):
                return i
        raise ValueError("Point not found in training data")
