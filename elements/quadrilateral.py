import numpy as np
from .element import Element


class Quadrilateral(Element):
    def value(self, p: np.ndarray) -> np.ndarray:
        s, t = p
        return 0.25 * np.array(
            [
                (1.0 - s) * (1.0 - t),
                (1.0 + s) * (1.0 - t),
                (1.0 + s) * (1.0 + t),
                (1.0 - s) * (1.0 + t),
            ]
        )

    def gradient(self, p: np.ndarray) -> np.ndarray:
        s, t = p
        # fmt: off
        return 0.25 * np.array([
            [-1.0 + t, -1.0 + s],
            [ 1.0 - t, -1.0 - s],
            [ 1.0 + t,  1.0 + s],
            [-1.0 - t,  1.0 - s],
        ])
        # fmt: on
