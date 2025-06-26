import numpy as np
from .element import Element


class Triangle(Element):
    def value(self, p: np.ndarray) -> np.ndarray:
        # If p.shape == (2, n)
        # then value(p) == (3, n)
        s, t = p
        # fmt: off
        return np.array([
            1 - s - t,
            s,
            t
        ])
        # fmt: on

    def gradient(self, p: np.ndarray) -> np.ndarray:
        # If p.shape == (2, n)
        # then gradient(p) == (3, 2, n)
        shape = p[0].shape
        o = np.ones(shape)
        z = np.zeros(shape)
        # fmt: off
        return np.array([
            [-o, -o],
            [ o,  z],
            [ z,  o],
        ])
        # fmt: on
