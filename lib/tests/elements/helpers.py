import numpy as np
from elements.element import Element

# fmt: off
LINE_SAMPLE_POINTS = np.array([
    -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
])[:, np.newaxis]
# fmt: on

# fmt: off
LINE_SAMPLE_DIRS = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
])[:, np.newaxis]
# fmt: on


def test_element_grad(el: Element, ps: np.ndarray, dirs: np.ndarray, eps: float = 1e-8):
    v = el.value(np.vstack([ps, ps + eps * dirs]).T)
    g = el.gradient(ps.T)
    points = ps.shape[0]
    el_order = v.shape[0]
    for k in range(el_order):
        actual = (v[k][points:] - v[k][:points]) / eps
        desired = np.sum(g[k].T * dirs, axis=1)
        np.testing.assert_allclose(actual, desired, atol=1e-7)


def make_2d_dirs(angles) -> np.ndarray:
    angles = np.asarray(angles) * np.pi / 180.0
    dirs = np.vstack([np.cos(angles), np.sin(angles)]).T
    return dirs


def test_base_points(el: Element, points: np.ndarray):
    v = el.value(points.T)
    np.testing.assert_allclose(v, np.eye(points.shape[0]))
