import unittest
import numpy as np
from meshio import Mesh
from mesh.boundary import mesh_boundary, mesh_prune_points


class BoundaryTests(unittest.TestCase):
    def test_rect_boundary(self):
        mesh = Mesh(
            [
                [0.0, 0.0],
                [1.25, 0.0],
                [2.5, 0.0],
                [3.75, 0.0],
                [5.0, 0.0],
                [0.0, 2.0],
                [1.25, 2.0],
                [2.5, 2.0],
                [3.75, 2.0],
                [5.0, 2.0],
                [0.0, 4.0],
                [1.25, 4.0],
                [2.5, 4.0],
                [3.75, 4.0],
                [5.0, 4.0],
            ],
            [
                (
                    "triangle",
                    [
                        [0, 1, 6],
                        [1, 2, 7],
                        [2, 3, 8],
                        [3, 4, 9],
                        [5, 6, 11],
                        [6, 7, 12],
                        [7, 8, 13],
                        [8, 9, 14],
                        [0, 6, 5],
                        [1, 7, 6],
                        [2, 8, 7],
                        [3, 9, 8],
                        [5, 11, 10],
                        [6, 12, 11],
                        [7, 13, 12],
                        [8, 14, 13],
                    ],
                )
            ],
        )

        bmesh = mesh_boundary(mesh)
        self.assertEqual(len(bmesh.cells), 1)
        self.assertEqual(bmesh.cells[0].type, "line")
        np.testing.assert_array_equal(
            bmesh.cells[0].data,
            [
                [0, 1],
                [5, 0],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 9],
                [10, 5],
                [9, 14],
                [11, 10],
                [12, 11],
                [13, 12],
                [14, 13],
            ],
        )

        umesh = mesh_prune_points(bmesh)
        self.assertEqual(len(umesh.cells), 1)
        self.assertEqual(umesh.cells[0].type, "line")
        np.testing.assert_array_equal(
            umesh.points,
            [
                [0.0, 0.0],
                [1.25, 0.0],
                [2.5, 0.0],
                [3.75, 0.0],
                [5.0, 0.0],
                [0.0, 2.0],
                [5.0, 2.0],
                [0.0, 4.0],
                [1.25, 4.0],
                [2.5, 4.0],
                [3.75, 4.0],
                [5.0, 4.0],
            ],
        )
        np.testing.assert_array_equal(
            umesh.cells[0].data,
            [
                [0, 1],
                [5, 0],
                [1, 2],
                [2, 3],
                [3, 4],
                [4, 6],
                [7, 5],
                [6, 11],
                [8, 7],
                [9, 8],
                [10, 9],
                [11, 10],
            ],
        )
        np.testing.assert_array_equal(
            umesh.point_data["point_idx"], [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14]
        )
