from collections.abc import Callable
from numpy import float64, abs, newaxis, prod, ndarray, sum, zeros, add
from numpy.linalg import det as matrix_det, svd as matrix_svd
from meshio import Mesh
from elements.element import Element
from elements.triangle import Triangle
from quadrature.base import BaseQuadrature
from quadrature.triangle import TriangleQuadrature


def grad_dot_grad(g1: ndarray, g2: ndarray) -> ndarray:
    return sum(g1 * g2, axis=0)


def _linear_for_element_type(
    points: ndarray,
    elements: ndarray,
    element: Element,
    quadrature: BaseQuadrature,
    linear_fn: Callable[[ndarray, ndarray], ndarray],
) -> ndarray:
    quad_points, quad_weights = quadrature.points_and_weights()

    _element_count, element_order = elements.shape
    point_count, point_dim = points.shape
    _quad_point_count, element_dim = quad_points.shape

    element_points = points[elements]
    # assert element_points.shape == (_element_count, element_order, point_dim)
    values_local = element.value(quad_points.T)
    # assert values_local.shape == (element_order, _quad_point_count)
    grads_local = element.gradient(quad_points.T)[:, :, newaxis, :].T
    # assert grads.shape == (_quad_point_count, 1, element_dim, element_order)

    coords = (values_local.T @ element_points).T
    # assert coords.shape == (point_dim, _quad_point_count, _element_count)
    J = grads_local @ element_points[newaxis, :, :, :]
    # assert J.shape == (_quad_point_count, _element_count, element_dim, point_dim)

    if point_dim > element_dim:
        # U, s, Vh = svd(J, full_matrices=False)
        # gradients = matrix_transpose(Vh) @ (
        #     (matrix_transpose(U) @ grads_local) / s[:, :, :, newaxis]
        # )
        s = matrix_svd(J, full_matrices=False, compute_uv=False)
        jacobians = prod(s, axis=2)
    else:
        # gradients = np_solve(J, grads_local)
        jacobians = abs(matrix_det(J))

    b = zeros((point_count,), dtype=float64)

    for i in range(0, element_order):
        n = elements[:, i]
        vals = values_local[i, :]
        r = linear_fn(vals[:, newaxis], coords)
        v = quad_weights @ (r * jacobians)
        add.at(b, n, v)

    return b


def from_linear(
    mesh: Mesh, linear_fn: Callable[[ndarray, ndarray], ndarray], norder=1
) -> ndarray:
    points = mesh.points
    b = zeros((points.shape[0],), dtype=float64)

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            element = Triangle()
            quadrature = TriangleQuadrature(norder)
        else:
            raise RuntimeError(f"Unsupported cell block {cell_block}")

        b += _linear_for_element_type(
            points, cell_block.data, element, quadrature, linear_fn
        )

    return b
