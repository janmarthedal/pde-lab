import unittest
import numpy as np
from pardeqsi.elements.line import Line
from .helpers import (
    test_element_grad,
    test_base_points,
    LINE_SAMPLE_POINTS,
    LINE_SAMPLE_DIRS,
)


class LineTests(unittest.TestCase):
    # fmt: off
    BASE_POINTS = np.array([
        [-1.0],
        [1.0],
    ])
    # fmt: on

    def test_base_points(self):
        test_base_points(Line(), self.BASE_POINTS)

    def test_grad(self):
        test_element_grad(Line(), LINE_SAMPLE_POINTS, LINE_SAMPLE_DIRS)


if __name__ == "__main__":
    unittest.main()
