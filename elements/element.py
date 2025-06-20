from abc import ABCMeta, abstractmethod

class Element(metaclass=ABCMeta):

    @abstractmethod
    def eval(self, p):
        return p

    @abstractmethod
    def grad(self, p):
        return p
