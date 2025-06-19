from abc import ABCMeta, abstractmethod

class Element(metaclass=ABCMeta):

    @property
    @abstractmethod
    def order(self):
        return 0

    @abstractmethod
    def eval(self, p):
        return p

    @abstractmethod
    def grad(self, p):
        return p
