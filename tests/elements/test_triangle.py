import unittest
import numpy as np
from elements.element import Element
from elements.triangle import Triangle


def test_element_grad(el: Element, ps: np.ndarray, dirs: np.ndarray, eps: float = 1e-8):
    v = el.eval(np.vstack([ps, ps + eps * dirs]).T)
    g = el.grad(ps.T)
    points = ps.shape[0]
    el_order = v.shape[0]
    for k in range(el_order):
        actual = (v[k][points:] - v[k][:points]) / eps
        desired = np.sum(g[k].T * dirs, axis=1)
        np.testing.assert_allclose(actual, desired, atol=1e-8)


def make_2d_dirs(angles) -> np.ndarray:
    angles = np.asarray(angles) * np.pi / 180.0
    dirs = np.vstack([np.cos(angles), np.sin(angles)]).T
    return dirs


def test_base_points(el: Element, points: np.ndarray):
    v = el.eval(points.T)
    np.testing.assert_allclose(v, np.eye(points.shape[0]))


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
