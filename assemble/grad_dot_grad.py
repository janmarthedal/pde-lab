from numpy import array, float64, dot, abs, zeros, empty, int32
from meshio import Mesh
from scipy.linalg import inv, det, solve
from scipy.sparse import coo_array, csr_array, diags_array
from elements.element import Element
from integrators.base_integrator import BaseIntegrator
from elements.triangle import Triangle
from integrators.triangle0_integrator import Triangle0Integrator


def element_grad_dot_grad(
    element_points: array,
    element: Element,
    integrator: BaseIntegrator,
    i: int,
    j: int,
) -> float:
    def integrand(p):
        element_grads = element.grad(p).T
        J = element_grads @ element_points
        element_grads_ij = inv(J) @ element_grads[:, (i, j)]
        # element_grads_ij = solve(J, element_grads[:, (i, j)])
        return abs(det(J)) * dot(element_grads_ij[:, 0], element_grads_ij[:, 1])

    v = integrator.integrate(integrand)
    return v


def element_assemble_grad_dot_grad(
    points: array, elements: array, element: Element, integrator: BaseIntegrator
) -> coo_array:
    order = elements.shape[1]
    dof = points.shape[0]
    coord_len = elements.shape[0] * (order - 1) * order // 2

    diag = zeros((dof,), dtype=float64)
    coords = empty((2, coord_len), dtype=int32)
    data = empty((coord_len,), dtype=float64)

    idx = 0
    for element_idx in elements:
        element_points = points[element_idx, :]

        for i in range(0, order):
            v = element_grad_dot_grad(element_points, element, integrator, i, i)
            diag[element_idx[i]] += v

            for j in range(0, i):
                v = element_grad_dot_grad(element_points, element, integrator, i, j)
                coords[:, idx] = (element_idx[i], element_idx[j])
                data[idx] = v
                idx += 1

    coords.sort(axis=0)
    d = diags_array(diag, offsets=0)
    a = coo_array((data, coords), shape=(dof, dof), dtype=float64).tocsr()
    return a + d + a.T


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
