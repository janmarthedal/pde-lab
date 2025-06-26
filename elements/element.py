from abc import ABCMeta, abstractmethod
import numpy as np


class Element(metaclass=ABCMeta):
    @abstractmethod
    def value(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
