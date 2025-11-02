import numpy as np
from typing import override
from .base import BaseQuadrature


class LineQuadrature(BaseQuadrature):
    def __init__(self, norder: int):
        self.norder: int = norder

    @override
    def points_and_weights(self) -> tuple[np.ndarray, np.ndarray]:
        if self.norder == 1:
            return (np.array([[0.5]]), np.array([1.0]))
        raise NotImplementedError(f"Line quadrature order {self.norder} not supported")
