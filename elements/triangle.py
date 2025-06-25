from numpy import array, ones, zeros
from .element import Element


class Triangle(Element):
    def eval(self, p):
        # If p.shape == (2, n)
        # then eval(p) == (3, n)
        s, t = p
        # fmt: off
        return array([
            1 - s - t,
            s,
            t
        ])
        # fmt: on

    def grad(self, p):
        # If p.shape == (2, n)
        # then grad(p) == (3, 2, n)
        shape = p[0].shape
        o = ones(shape)
        z = zeros(shape)
        # fmt: off
        return array([
            [-o, -o],
            [ o,  z],
            [ z,  o]
        ])
        # fmt: on
