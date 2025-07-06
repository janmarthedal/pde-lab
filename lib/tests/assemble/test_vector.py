import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from pardeqsi.mesh.generate import rect2d
from pardeqsi.assemble.vector import from_linear


def linear_fn(vals: np.ndarray, coords: np.ndarray):
    return np.prod(np.sin(np.pi * coords), axis=0) * vals


def linear_fn_yz(vals: np.ndarray, coords: np.ndarray):
    return (np.sin(np.pi * coords[1]) * np.sin(np.pi * coords[2])) * vals


class VectorTests(unittest.TestCase):
    brect = np.array(
        [
            3.1212924306639023e-01,
            -3.6823568531065498e-01,
            3.4853891342949683e-01,
            -1.2467277307612086e-01,
            1.3990425617980798e-01,
            1.7222498688658283e-01,
            -2.4356291223453391e-01,
            3.7470027081099033e-16,
            2.4356291223453444e-01,
            -1.7222498688658205e-01,
            -1.3990425617980748e-01,
            1.2467277307612118e-01,
            -3.4853891342949639e-01,
            3.6823568531065498e-01,
            -3.1212924306638995e-01,
        ]
    )

    def test_rectangle(self):
        mesh = rect2d(5, 4, 5, 3)
        b = from_linear(mesh, linear_fn, norder=2)
        np.testing.assert_array_almost_equal(b, self.brect)

    def test_rectangle_yz(self):
        mesh = rect2d(5, 4, 5, 3, addz=True)
        mesh.points = mesh.points @ np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]
        )
        b = from_linear(mesh, linear_fn_yz, norder=2)
        np.testing.assert_array_almost_equal(b, self.brect)

    def test_rectangle_3d(self):
        mesh = rect2d(5, 4, 5, 3, addz=True)
        r = Rotation.from_euler("zyx", [90, 45, 30], degrees=True)
        mesh.points = r.apply(mesh.points)
        b = from_linear(mesh, linear_fn, norder=2)
        np.testing.assert_array_almost_equal(
            b,
            np.array(
                [
                    -0.0897993795609674,
                    -0.1839033322200594,
                    0.1364884449842456,
                    0.3422350713237634,
                    0.0124610886943806,
                    0.4035757373548856,
                    0.2404170007313487,
                    -0.3151415254101054,
                    -0.2777990053352862,
                    0.2232983704832084,
                    -0.0157080766384685,
                    0.5310728932312703,
                    0.2633738578606202,
                    -0.3192663378276079,
                    -0.2999274401311097,
                ]
            ),
            err_msg="Regression error (result not verified)",
        )


if __name__ == "__main__":
    unittest.main()
