import unittest
from pardeqsi.mesh.generate import rect2d
from pardeqsi.assemble.matrix import from_bilinear, grad_dot_grad
from numpy.testing import assert_array_almost_equal
from scipy.sparse import coo_array
from scipy.spatial.transform import Rotation


class MatrixTests(unittest.TestCase):
    # fmt: off
    Arect = coo_array(([
        1.1125, -0.8, -0.3125, -0.8, 2.225, -0.8, -0.625, -0.8, 2.225, -0.8, -0.625,
        -0.8, 2.225, -0.8, -0.625, -0.8, 1.1125, -0.3125, -0.3125, 2.225, -1.6,
        -0.3125, -0.625, -1.6, 4.45, -1.6, -0.625, -0.625, -1.6, 4.45, -1.6, -0.625,
        -0.625, -1.6, 4.45, -1.6, -0.625, -0.3125, -1.6, 2.225, -0.3125, -0.3125,
        1.1125, -0.8, -0.625, -0.8, 2.225, -0.8, -0.625, -0.8, 2.225, -0.8, -0.625,
        -0.8, 2.225, -0.8, -0.3125, -0.8, 1.1125
    ], ([
        0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
        6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11, 12,
        12, 12, 12, 13, 13, 13, 13, 14, 14, 14
    ], [
        0, 1, 5, 0, 1, 2, 6, 1, 2, 3, 7, 2, 3, 4, 8, 3, 4, 9, 0, 5, 6, 10, 1, 5, 6, 7,
        11, 2, 6, 7, 8, 12, 3, 7, 8, 9, 13, 4, 8, 9, 14, 5, 10, 11, 6, 10, 11, 12, 7,
        11, 12, 13, 8, 12, 13, 14, 9, 13, 14
    ])), shape=(15, 15)).toarray()
    # fmt: on

    def test_rectangle(self):
        mesh = rect2d(5, 4, 5, 3)
        A = from_bilinear(mesh, grad_dot_grad)
        assert_array_almost_equal(A.toarray(), self.Arect)

    def test_rectangle_3d(self):
        mesh = rect2d(5, 4, 5, 3, addz=True)
        r = Rotation.from_euler("zyx", [90, 45, 30], degrees=True)
        mesh.points = r.apply(mesh.points)
        A = from_bilinear(mesh, grad_dot_grad)
        assert_array_almost_equal(A.toarray(), self.Arect)


if __name__ == "__main__":
    unittest.main()
