from collections.abc import Callable
from numpy import float64, abs, newaxis, prod, ndarray, sum, matrix_transpose
from numpy.linalg import solve as np_solve, det as np_det, svd
from meshio import Mesh
from scipy.sparse import coo_array, csr_array
from elements.element import Element
from elements.triangle import Triangle
from quadrature.base import BaseQuadrature
from quadrature.triangle import TriangleQuadrature


def grad_dot_grad(g1: ndarray, g2: ndarray) -> ndarray:
    return sum(g1 * g2, axis=0)


def _bilinear_for_element_type(
    points: ndarray,
    elements: ndarray,
    element: Element,
    quadrature: BaseQuadrature,
    bilinear_fn: Callable[[ndarray, ndarray], ndarray],
) -> coo_array:
    quad_points, quad_weights = quadrature.points_and_weights()

    element_count, element_order = elements.shape
    point_count, point_dim = points.shape
    quad_point_count, element_dim = quad_points.shape

    element_points = points[elements]
    # assert element_points.shape == (element_count, element_order, point_dim)
    g = element.grad(quad_points.T)[:, :, newaxis, :].T
    # assert g.shape == (quad_point_count, 1, element_dim, element_order)

    J = g @ element_points[newaxis, :, :, :]
    # assert J.shape == (quad_point_count, element_count, element_dim, point_dim)

    if point_dim > element_dim:
        U, s, Vh = svd(J, full_matrices=False)
        grads = matrix_transpose(Vh) @ ((matrix_transpose(U) @ g) / s[:, :, :, newaxis])
        jacobians = prod(s, axis=2)
    else:
        grads = np_solve(J, g)
        jacobians = abs(np_det(J))

    R = csr_array((point_count, point_count), dtype=float64)

    for i in range(0, element_order):
        ni = elements[:, i]
        gi = grads[:, :, :, i].T
        for j in range(0, element_order):
            nj = elements[:, j]
            gj = grads[:, :, :, j].T
            b = bilinear_fn(gi, gj).T
            v = quad_weights @ (b * jacobians)
            R += coo_array(
                (v, (ni, nj)), shape=(point_count, point_count), dtype=float64
            )

    return R


def from_bilinear(
    mesh: Mesh, bilinear_fn: Callable[[ndarray, ndarray], ndarray], norder=1
) -> csr_array:
    points = mesh.points
    point_count = points.shape[0]
    A = csr_array((point_count, point_count), dtype=float64)

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            element = Triangle()
            quadrature = TriangleQuadrature(norder)
        else:
            raise RuntimeError(f"Unsupported cell block {cell_block}")

        A += _bilinear_for_element_type(
            points, cell_block.data, element, quadrature, bilinear_fn
        )

    return A
