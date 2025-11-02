import numpy as np
from typing import override
from .base import BaseQuadrature


class TriangleQuadrature(BaseQuadrature):
    MEASURE: float = 0.5

    def __init__(self, norder: int):
        self.norder: int = norder

    @override
    def points_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        if self.norder == 1:
            return (np.array([[1.0 / 3.0, 1.0 / 3.0]]), self.MEASURE * np.array([1.0]))
        if self.norder == 2:
            return (
                np.array(
                    [
                        [1.0 / 6.0, 1.0 / 6.0],
                        [1.0 / 6.0, 4.0 / 6.0],
                        [4.0 / 6.0, 1.0 / 6.0],
                    ]
                ),
                self.MEASURE * np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            )
        raise NotImplementedError(
            f"Triangle quadrature order {self.norder} not supported"
        )
