from abc import ABCMeta, abstractmethod


class BaseIntegrator(metaclass=ABCMeta):
    @abstractmethod
    def integrate(self, fun: callable) -> float:
        return 0.0
