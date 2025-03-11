from abc import ABC, abstractmethod

import numpy as np


class Scaler(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray) -> None:
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class MinMaxScaler(Scaler):
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X: np.ndarray) -> None:
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.min is None or self.max is None:
            raise ValueError("Scaler is not fitted yet")
        range_ = self.max - self.min
        range_[range_ == 0] = 1
        return (X - self.min) / range_


class StandardScaler(Scaler):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler is not fitted yet")
        return (X - self.mean) / self.std


class RobustScaler(Scaler):
    def __init__(self):
        self.median = None
        self.q1 = None
        self.q3 = None

    def fit(self, X: np.ndarray) -> None:
        self.median = np.median(X, axis=0)
        self.q1 = np.percentile(X, 25, axis=0)
        self.q3 = np.percentile(X, 75, axis=0)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.median is None or self.q1 is None or self.q3 is None:
            raise ValueError("Scaler is not fitted yet")

        iqr = self.q3 - self.q1

        iqr[iqr == 0] = 1

        return (X - self.median) / iqr
