from collections.abc import Callable
import numpy as np
from scipy.sparse import coo_array, csr_array
from mesh.mesh import Mesh
from elements.element import Element
from elements.triangle import Triangle
from quadrature.base import BaseQuadrature
from quadrature.triangle import TriangleQuadrature


def grad_dot_grad(g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
    return np.sum(g1 * g2, axis=0)


def _bilinear_for_element_type(
    points: np.ndarray,
    elements: np.ndarray,
    element: Element,
    quadrature: BaseQuadrature,
    bilinear_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> coo_array:
    quad_points, quad_weights = quadrature.points_and_weights()

    _element_count, element_order = elements.shape
    point_count, point_dim = points.shape
    _quad_point_count, element_dim = quad_points.shape

    element_points = points[elements]
    # assert element_points.shape == (_element_count, element_order, point_dim)
    grads_local = element.gradient(quad_points.T)[:, :, np.newaxis, :].T
    # assert grads_local.shape == (_quad_point_count, 1, element_dim, element_order)

    J = grads_local @ element_points[np.newaxis, :, :, :]
    # assert J.shape == (_quad_point_count, _element_count, element_dim, point_dim)

    if point_dim > element_dim:
        U, s, Vh = np.linalg.svd(J, full_matrices=False)
        gradients = np.matrix_transpose(Vh) @ (
            (np.matrix_transpose(U) @ grads_local) / s[:, :, :, np.newaxis]
        )
        jacobians = np.prod(s, axis=2)
    else:
        gradients = np.linalg.solve(J, grads_local)
        jacobians = np.abs(np.linalg.det(J))

    R = csr_array((point_count, point_count), dtype=np.float64)

    for i in range(0, element_order):
        ni = elements[:, i]
        gi = gradients[:, :, :, i].T
        for j in range(0, element_order):
            nj = elements[:, j]
            gj = gradients[:, :, :, j].T
            b = bilinear_fn(gi, gj).T
            v = quad_weights @ (b * jacobians)
            R += coo_array(
                (v, (ni, nj)), shape=(point_count, point_count), dtype=np.float64
            )

    return R


def from_bilinear(
    mesh: Mesh, bilinear_fn: Callable[[np.ndarray, np.ndarray], np.ndarray], norder=1
) -> csr_array:
    points = mesh.points
    A = csr_array((points.shape[0], points.shape[0]), dtype=np.float64)

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
