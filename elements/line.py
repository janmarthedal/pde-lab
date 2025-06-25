from numpy import array, ones_like
from .element import Element


class Line(Element):
    def eval(self, p):
        # If p.shape == (1, n)
        # then eval(p) == (2, n)
        s = p[0]
        # fmt: off
        return array([
            1 - s,
            s,
        ])
        # fmt: on

    def grad(self, p):
        # If p.shape == (1, n)
        # then grad(p) == (2, 1, n)
        o = ones_like(p[0].shape)
        # fmt: off
        return array([
            [-o],
            [ o],
        ])
        # fmt: on
