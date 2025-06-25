import unittest
import numpy as np
from elements.quadrilateral import Quadrilateral
from .test_triangle import test_element_grad, make_2d_dirs, test_base_points


class QuadrilateralTests(unittest.TestCase):
    # fmt: off
    BASE_POINTS = np.array([
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
    ])
    # fmt: on
    SAMPLE_POINTS = np.array(
        [
            [-1.0, -1.0],
            [-0.5, -1.0],
            [0.0, -1.0],
            [0.5, -1.0],
            [1.0, -1.0],
            [-1.0, -0.5],
            [-0.5, -0.5],
            [0.0, -0.5],
            [0.5, -0.5],
            [1.0, -0.5],
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [-1.0, 0.5],
            [-0.5, 0.5],
            [0.0, 0.5],
            [0.5, 0.5],
            [1.0, 0.5],
            [-1.0, 1.0],
            [-0.5, 1.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ]
    )
    SAMPLE_DIRS = make_2d_dirs(
        [
            0.0,
            10.0,
            20.0,
            30.0,
            100.0,
            5.0,
            15.0,
            25.0,
            35.0,
            45.0,
            55.0,
            65.0,
            75.0,
            85.0,
            95.0,
            105.0,
            115.0,
            125.0,
            135.0,
            145.0,
            0.0,
            -10.0,
            -20.0,
            -30.0,
            -100.0,
        ]
    )

    def test_base_points(self):
        test_base_points(Quadrilateral(), self.BASE_POINTS)

    def test_grad(self):
        test_element_grad(
            Quadrilateral(), QuadrilateralTests.SAMPLE_POINTS, QuadrilateralTests.SAMPLE_DIRS
        )


if __name__ == "__main__":
    unittest.main()
