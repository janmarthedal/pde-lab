from typing import override
import numpy as np
from .element import Element


class Line3(Element):
    @override
    def value(self, p: np.ndarray) -> np.ndarray:
        # If p.shape == (1, n)
        # then value(p) == (3, n)
        s = p[0]
        # fmt: off
        return np.array([
            0.5 * (s - 1.0) * s,
            (1.0 - s) * (1.0 + s),
            0.5 * s * (1.0 + s),
        ])
        # fmt: on

    @override
    def gradient(self, p: np.ndarray) -> np.ndarray:
        # If p.shape == (1, n)
        # then gradient(p) == (2, 1, n)
        s = p[0]
        # fmt: off
        return np.array([
            [ s - 0.5],
            [-2.0 * s],
            [ s + 0.5],
        ])
        # fmt: on
