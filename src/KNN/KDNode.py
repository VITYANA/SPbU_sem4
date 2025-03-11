from dataclasses import dataclass
from math import inf
from typing import Generic, List, Optional, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


@dataclass
class Point(Generic[T]):
    coord: NDArray[np.float64]

    def __lt__(self, other):
        if isinstance(other, Point):
            return np.all(self.coord < other.coord)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(self.coord))

    def __eq__(self, other):
        if isinstance(other, Point):
            return np.allclose(self.coord, other.coord)
        return NotImplemented


class KDNode(Generic[T]):
    def __init__(self, X: List[Point[T]], leaf_size: int):
        if leaf_size <= 0:
            raise AttributeError("Leaf size must be strictly positive")

        if not X:
            raise ValueError("List of points cannot be empty")

        self.axis: int = 0
        self.axis_median: float = 0
        self.is_leaf: bool = False
        self.left: Optional[KDNode[T]] = None
        self.right: Optional[KDNode[T]] = None

        if len(X) > leaf_size:
            self.axis = self._choose_axis(X)
            axis_median, left_X, right_X = self._split_axis(X, self.axis)

            self.axis_median = float(axis_median)

            self.left = KDNode(left_X, leaf_size)
            self.right = KDNode(right_X, leaf_size)

        else:
            self.is_leaf = True
            self.X = X

    @staticmethod
    def _choose_axis(X: List[Point[T]]) -> int:
        output_axis = 0
        max_spread = -inf

        for axis in range(len(X[0].coord)):
            min_in_axis, max_in_axis = inf, -inf

            for x in X:
                max_in_axis = max(max_in_axis, x.coord[axis].item())
                min_in_axis = min(min_in_axis, x.coord[axis].item())

            spread = max_in_axis - min_in_axis
            if max_spread < spread:
                max_spread = spread
                output_axis = axis

        return output_axis

    @staticmethod
    def _split_axis(
        X: List[Point[T]], axis: int
    ) -> tuple[np.float64, list[Point[T]], list[Point[T]]]:
        left_axis = []
        right_axis = []
        axis_median = np.float64(np.median([x.coord[axis] for x in X]))

        for x in X:
            if x.coord[axis] < axis_median:
                left_axis.append(x)
            else:
                right_axis.append(x)

        if not left_axis or not right_axis:
            left_axis, right_axis = X[: len(X) // 2], X[len(X) // 2 :]

        return axis_median, left_axis, right_axis
