from numpy import array, ones_like, ndarray
from .element import Element


class Line(Element):
    def value(self, p: ndarray) -> ndarray:
        # If p.shape == (1, n)
        # then value(p) == (2, n)
        s = p[0]
        # fmt: off
        return array([
            0.5 * (1.0 - s),
            0.5 * (1.0 + s),
        ])
        # fmt: on

    def gradient(self, p: ndarray) -> ndarray:
        # If p.shape == (1, n)
        # then gradient(p) == (2, 1, n)
        o = ones_like(p[0].shape)
        # fmt: off
        return array([
            [-0.5 * o],
            [ 0.5 * o],
        ])
        # fmt: on
