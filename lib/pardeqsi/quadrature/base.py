from abc import ABCMeta, abstractmethod
import numpy as np


class BaseQuadrature(metaclass=ABCMeta):
    @abstractmethod
    def points_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
