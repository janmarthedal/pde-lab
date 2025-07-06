import unittest
import numpy as np
from elements.triangle import Triangle
from .helpers import test_element_grad, make_2d_dirs, test_base_points


class TriangleTests(unittest.TestCase):
    # fmt: off
    BASE_POINTS = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])
    # fmt: on
    SAMPLE_POINTS = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
            [0.0, 0.5],
            [1.0 / 3.0, 1.0 / 3.0],
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 6.0],
            [0.4, 0.3],
            [0.2, 0.7],
        ]
    )
    SAMPLE_DIRS = make_2d_dirs(
        [0.0, 10.0, 180.0, -45.0, -90.0, 20.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    )

    def test_base_points(self):
        test_base_points(Triangle(), self.BASE_POINTS)

    def test_grad(self):
        test_element_grad(Triangle(), self.SAMPLE_POINTS, self.SAMPLE_DIRS)


if __name__ == "__main__":
    unittest.main()
