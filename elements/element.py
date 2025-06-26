from abc import ABCMeta, abstractmethod
from numpy import ndarray


class Element(metaclass=ABCMeta):
    @abstractmethod
    def value(self, p: ndarray) -> ndarray:
        raise NotImplementedError()

    @abstractmethod
    def gradient(self, p: ndarray) -> ndarray:
        raise NotImplementedError()
