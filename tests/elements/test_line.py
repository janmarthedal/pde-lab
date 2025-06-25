import unittest
import numpy as np
from elements.line import Line
from .helpers import test_element_grad, test_base_points


class LineTests(unittest.TestCase):
    # fmt: off
    BASE_POINTS = np.array([
        [0.0],
        [1.0],
    ])
    # fmt: on
    SAMPLE_POINTS = np.array(
        [[0.0], [0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [1.0]]
    )
    SAMPLE_DIRS = np.array(
        [[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
    )

    def test_base_points(self):
        test_base_points(Line(), self.BASE_POINTS)

    def test_grad(self):
        test_element_grad(Line(), self.SAMPLE_POINTS, self.SAMPLE_DIRS)


if __name__ == "__main__":
    unittest.main()
