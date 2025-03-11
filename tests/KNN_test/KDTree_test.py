import os
import random
import sys

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.KNN.KDNode import Point, T
from src.KNN.KDTree import KDTree


def brute_force_search(
    train: list[Point[T]], test: list[Point[T]], k: int
) -> dict[Point[T], list[Point[T]]]:
    knn = {}
    for test_point in test:
        neighbors = []
        for point in train:
            dist = KDTree.distance(test_point, point)
            neighbors.append((point, dist))

        neighbors = sorted(neighbors, key=lambda x: x[1])
        knn[test_point] = [p[0] for p in neighbors[:k]]

    return knn


class TestKDTree:
    @given(
        st.integers(min_value=100, max_value=200),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=10),
        st.integers(min_value=1, max_value=30),
    )
    def test_query(self, train_size, k, leaf_size, neighbours):
        x_train = [
            Point(np.array([random.randint(-100, 100) for _ in range(k)]))
            for _ in range(train_size)
        ]
        x_test = [
            Point(np.array([random.randint(-100, 100) for _ in range(k)]))
            for _ in range(30)
        ]
        kdtree = KDTree(x_train, leaf_size)

        tree_search = kdtree.query(x_test, neighbours)
        stupid_search = brute_force_search(x_train, x_test, neighbours)

        for point in x_test:
            stupid_dist = sorted(
                [KDTree.distance(point, near_point) for near_point in stupid_search]
            )
            tree_dist = sorted(
                [KDTree.distance(point, near_point) for near_point in tree_search]
            )

            for i in range(min(neighbours, len(stupid_dist), len(tree_dist))):
                assert np.isclose(stupid_dist[i], tree_dist[i]), (
                    stupid_dist[i],
                    tree_dist[i],
                )

    def test_empty_tree(self):
        x_train = []
        with pytest.raises(ValueError):
            kdtree = KDTree(x_train, leaf_size=10)

    def test_single_point_tree(self):
        x_train = [Point(np.array([1, 2, 3]))]
        x_test = [Point(np.array([1, 2, 3]))]
        kdtree = KDTree(x_train, leaf_size=10)
        result = kdtree.query(x_test, k=1)
        assert (
            result[x_test[0]] == x_train
        ), "Single point tree should return the same point for any query"

    def test_k_greater_than_train_size(self):
        x_train = [Point(np.array([1, 2, 3])), Point(np.array([4, 5, 6]))]
        x_test = [Point(np.array([1, 2, 3]))]
        kdtree = KDTree(x_train, leaf_size=10)
        result = kdtree.query(x_test, k=3)
        assert (
            len(result[x_test[0]]) == 2
        ), "Should return all points when k is greater than train size"

    def test_zero_k(self):
        x_train = [Point(np.array([1, 2, 3])), Point(np.array([4, 5, 6]))]
        x_test = [Point(np.array([1, 2, 3]))]
        kdtree = KDTree(x_train, leaf_size=10)
        with pytest.raises(ValueError):
            kdtree.query(x_test, k=0)

    def test_negative_k(self):
        x_train = [Point(np.array([1, 2, 3])), Point(np.array([4, 5, 6]))]
        x_test = [Point(np.array([1, 2, 3]))]
        kdtree = KDTree(x_train, leaf_size=10)
        with pytest.raises(ValueError):
            kdtree.query(x_test, k=-1)

    def test_same_points(self):
        x_train = [Point(np.array([1, 2, 3])), Point(np.array([1, 2, 3]))]
        x_test = [Point(np.array([1, 2, 3]))]
        kdtree = KDTree(x_train, leaf_size=10)
        result = kdtree.query(x_test, k=2)
        assert (
            len(result[x_test[0]]) == 2
        ), "Should return all points even if they are the same"
