from numpy import array
from .element import Element


class Triangle(Element):

    _grad = array([
            [-1., -1.],
            [ 1.,  0.],
            [ 0.,  1.]
        ])

    def eval(self, p):
        r = self._grad @ p
        r[0] += 1.0
        return r

    def grad(self, _p):
        return self._grad
