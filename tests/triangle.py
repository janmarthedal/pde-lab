import unittest
from numpy import array
from numpy.testing import assert_array_almost_equal
from elements.triangle import Triangle


class ElementTests(unittest.TestCase):

    def test_triangle(self):
        e = Triangle()
        r = e.eval(array([1.0, 2.0]))
        assert_array_almost_equal(r, array([-2.0, 1.0, 2.0]))
        r = e.eval(array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).T)
        assert_array_almost_equal(
            r, array([[-2.0, 1.0, 2.0], [-6.0, 3.0, 4.0], [-10.0, 5.0, 6.0]]).T
        )


if __name__ == "__main__":
    unittest.main()
