from numpy import array
from .element import Element


class Line3(Element):
    def eval(self, p):
        # If p.shape == (1, n)
        # then eval(p) == (3, n)
        s = p[0]
        # fmt: off
        return array([
            0.5 * (s - 1.0) * s,
            (1.0 - s) * (1.0 + s),
            0.5 * s * (1.0 + s),
        ])
        # fmt: on

    def grad(self, p):
        # If p.shape == (1, n)
        # then grad(p) == (2, 1, n)
        s = p[0]
        # fmt: off
        return array([
            [ s - 0.5],
            [-2.0 * s],
            [ s + 0.5],
        ])
        # fmt: on
