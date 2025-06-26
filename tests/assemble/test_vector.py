import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal
from mesh.generate import rect2d
from assemble.vector import from_linear


def linear_fn(vals: np.ndarray, coords: np.ndarray):
    return np.prod(np.sin(np.pi * coords), axis=0) * vals


class VectorTests(unittest.TestCase):
    brect = np.array(
        [
            3.12129243e-01,
            -3.68235685e-01,
            3.48538913e-01,
            -1.24672773e-01,
            1.39904256e-01,
            1.72224987e-01,
            -2.43562912e-01,
            3.74700271e-16,
            2.43562912e-01,
            -1.72224987e-01,
            -1.39904256e-01,
            1.24672773e-01,
            -3.48538913e-01,
            3.68235685e-01,
            -3.12129243e-01,
        ]
    )

    def test_rectangle(self):
        mesh = rect2d(5, 4, 5, 3)
        b = from_linear(mesh, linear_fn, norder=2)
        assert_array_almost_equal(b, self.brect)


if __name__ == "__main__":
    unittest.main()
