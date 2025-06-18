from numpy import array, float64, dot, abs
from meshio import Mesh
from scipy.linalg import inv, det
from scipy.sparse import coo_array, csr_array
from elements.base_element import BaseElement
from integrators.base_integrator import BaseIntegrator
from elements.triangle import Triangle
from integrators.triangle0_integrator import Triangle0Integrator


def element_grad_dot_grad(element_points: array, element: BaseElement, integrator: BaseIntegrator, i: int, j: int) -> float:
    def integrand(p):
        element_grads = element.grad(p)
        J = element_grads.T @ element_points
        Jinv = inv(J)
        return abs(det(J)) * dot(Jinv @ element_grads[i], Jinv @ element_grads[j])
    # def integrand(s, t):
    #     bg = elem_grad(np.array([s, t]))
    #     J = bg.T @ element_points
    #     U, s, _ = svd(J)
    #     jacobian = np.abs(np.prod(s))
    #     sinv = np.reciprocal(s)
    #     return jacobian * np.dot(sinv * (U.T @ bg[i]), sinv * (U.T @ bg[j]))
    v = integrator.integrate(integrand)
    return v



def element_assemble_grad_dot_grad(points: array, elements: array, element: BaseElement, integrator: BaseIntegrator) -> coo_array:
    order = elements.shape[1]

    # pre-allocate np arrays instead?
    row: list[float] = []
    col: list[float] = []
    data: list[float] = []

    for element_idx in elements:
        element_points = points[element_idx, :]

        for i in range(0, order):
            for j in range(0, order):
                # exploit symmetry?
                v = element_grad_dot_grad(element_points, element, integrator, i, j)
                row.append(element_idx[i])
                col.append(element_idx[j])
                data.append(v)

    return coo_array(
        (data, (row, col)),
        shape=(points.shape[0], points.shape[0]),
        dtype=float64
    ).tocsr()


def assemble_grad_dot_grad(mesh: Mesh) -> csr_array:
    points = mesh.points
    dim = points.shape[0]
    A = csr_array((dim, dim), dtype=float64)

    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            element = Triangle()
            integrator = Triangle0Integrator()
            A += element_assemble_grad_dot_grad(points, cell_block.data, element, integrator)
        else:
            print(f"Unsupported cell block {cell_block}")

    return A
