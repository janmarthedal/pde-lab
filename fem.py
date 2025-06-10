import numpy as np

class Triangle:

    def bilinear_prod(self, P, i, j):
        assert P.shape == (3, 3)


triangle = Triangle()

points = np.array([
    [1, 2, 0],
    [5, 1, 0],
    [3, 5, 0],
])

triangle.bilinear_prod(points, 0, 0)
