from abc import ABCMeta, abstractmethod
from numpy import ndarray


class BaseQuadrature(metaclass=ABCMeta):
    @abstractmethod
    def points_and_weights(self) -> tuple[ndarray, ndarray]:
        raise NotImplementedError()
