from numpy import array, float64, abs, newaxis, prod, ndarray, sum, matrix_transpose
from numpy.linalg import solve as np_solve, det as np_det, svd
from meshio import Mesh
from scipy.sparse import coo_array, csr_array
from elements.element import Element
from integrators.base_integrator import BaseIntegrator
from elements.triangle import Triangle
from integrators.triangle0_integrator import Triangle0Integrator


def bilinear_fn(g1: ndarray, g2: ndarray):
    return sum(g1 * g2, axis=0)


def element_assemble_grad_dot_grad(
    points: ndarray, elements: ndarray, element: Element, integrator: BaseIntegrator
) -> coo_array:
    # quad_points = array([[2./3., 1./6.], [1./3., 2./6.], [1./3., 1./6.]])
    # quad_weights = 0.5 * array([1./3., 1./3., 1./3.])
    quad_points = array([[0.0, 0.0]])
    quad_weights = 0.5 * array([1.0])

    element_count, element_order = elements.shape
    point_count, point_dim = points.shape
    quad_point_count, element_dim = quad_points.shape

    element_points = points[elements]
    assert element_points.shape == (element_count, element_order, point_dim)
    g = matrix_transpose(element.grad(quad_points))[:, newaxis, :, :]
    assert g.shape == (quad_point_count, 1, element_dim, element_order)

    J = g @ element_points[newaxis, :, :, :]
    assert J.shape == (quad_point_count, element_count, element_dim, point_dim)

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


def assemble_grad_dot_grad(mesh: Mesh) -> csr_array:
    points = mesh.points
    dim = points.shape[0]
    A = csr_array((dim, dim), dtype=float64)

    for cell_block in mesh.cells:
        if cell_block.type == "triangle":
            element = Triangle()
            integrator = Triangle0Integrator()
            A += element_assemble_grad_dot_grad(
                points, cell_block.data, element, integrator
            )
        else:
            print(f"Unsupported cell block {cell_block}")

    return A
