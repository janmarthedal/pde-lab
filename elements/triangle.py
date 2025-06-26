from numpy import array, ones, zeros, ndarray
from .element import Element


class Triangle(Element):
    def value(self, p: ndarray) -> ndarray:
        # If p.shape == (2, n)
        # then value(p) == (3, n)
        s, t = p
        # fmt: off
        return array([
            1 - s - t,
            s,
            t
        ])
        # fmt: on

    def gradient(self, p: ndarray) -> ndarray:
        # If p.shape == (2, n)
        # then gradient(p) == (3, 2, n)
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
