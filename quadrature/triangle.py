from numpy import array, ndarray
from .base import BaseQuadrature


class TriangleQuadrature(BaseQuadrature):
    measure = 0.5

    def __init__(self, norder: int):
        self.norder = norder

    def points_and_weights(self) -> tuple[ndarray, ndarray]:
        if self.norder == 1:
            return (array([[0.0, 0.0]]), TriangleQuadrature.measure * array([1.0]))
        if self.norder == 2:
            return (
                array(
                    [
                        [2.0 / 3.0, 1.0 / 6.0],
                        [1.0 / 3.0, 2.0 / 6.0],
                        [1.0 / 3.0, 1.0 / 6.0],
                    ]
                ),
                TriangleQuadrature.measure * array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
            )
        raise NotImplementedError(
            f"Triangle quadrature order {self.norder} not supported"
        )
