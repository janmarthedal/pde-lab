import unittest
import numpy as np
from elements.line3 import Line3
from .helpers import test_element_grad, test_base_points, LINE_SAMPLE_POINTS, LINE_SAMPLE_DIRS


class Line3Tests(unittest.TestCase):
    # fmt: off
    BASE_POINTS = np.array([
        [-1.0],
        [ 0.0],
        [ 1.0],
    ])
    # fmt: on

    def test_base_points(self):
        test_base_points(Line3(), self.BASE_POINTS)

    def test_grad(self):
        test_element_grad(Line3(), LINE_SAMPLE_POINTS, LINE_SAMPLE_DIRS)


if __name__ == "__main__":
    unittest.main()
