from numpy import array, ones
from .element import Element


class Triangle(Element):
    # fmt: off
    _grad = array([
        [-1.0, -1.0],
        [ 1.0,  0.0],
        [ 0.0,  1.0]
    ])
    # fmt: on

    def eval(self, p):
        r = self._grad @ p
        r[0] += 1.0
        return r

    def grad(self, p):
        """
        p: A list of points, p.shape[1] == 2
        """
        return ones((p.shape[0], 1, 1)) * self._grad
