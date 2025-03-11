import heapq
from typing import Generic, List, Optional, Tuple, TypeVar

import numpy as np

from src.KNN.KDNode import KDNode, Point, T

V = TypeVar("V")


class KDTree(Generic[V]):

    def __init__(self, X: List[Point[T]], leaf_size: int):

        self._root: KDNode = KDNode(X, leaf_size)
        self._leaf_size: int = leaf_size

    def _k_nearest_neighbors(
        self, x: Point[T], k: int, current_node: KDNode[T]
    ) -> List[Point[T]]:
        neighbors: List[Tuple[float, Point[T]]] = []

        def search(node: Optional[KDNode[T]]):
            if node is None:
                return

            if node.is_leaf:
                for point in node.X:
                    distance = self.distance(x, point)
                    if len(neighbors) < k:
                        heapq.heappush(neighbors, (-distance, point))
                    else:
                        if -distance > neighbors[0][0]:
                            heapq.heappushpop(neighbors, (-distance, point))
                return

            axis = node.axis
            axis_median = node.axis_median

            if x.coord[axis] < axis_median:
                search(node.left)
                other_subtree = node.right
            else:
                search(node.right)
                other_subtree = node.left

            if (
                len(neighbors) < k
                or abs(x.coord[axis] - axis_median) < -neighbors[0][0]
            ):
                search(other_subtree)

        search(current_node)

        return [point for _, point in sorted(neighbors, key=lambda item: -item[0])]

    def query(self, X: List[Point[T]], k: int) -> dict[Point[T], list[Point[T]]]:
        if k <= 0:
            raise ValueError("k must be a positive integer")

        neighbors_for_point = {}
        for x in X:
            if self._root is None:
                neighbors_for_point[x] = []
            else:
                neighbors_for_point[x] = self._k_nearest_neighbors(x, k, self._root)
        return neighbors_for_point

    @staticmethod
    def distance(point1: Point[T], point2: Point[T]) -> float:
        if len(point1.coord) != len(point2.coord):
            raise ValueError("Points must have the same dimensionality")

        return float(np.linalg.norm(point1.coord - point2.coord))
