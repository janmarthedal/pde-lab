import time
import numpy as np
from meshio import Mesh
from meshgen import rect2d
from scipy.integrate import dblquad
from scipy.linalg import det, inv
from scipy.sparse import coo_array, csr_array


# def get_triangle_fun(i: int) -> Callable[[np.array], np.array]:
#     match i:
#         case 0: return lambda p: 1 - p[0] - p[1]
#         case 1: return lambda p: p[0]
#         case 2: return lambda p: p[1]
#     assert False


# def get_triangle_grad_fun(i: int) -> Callable[[np.array], np.array]:
#     match i:
#         case 0: return lambda _: np.array([-1., -1.])
#         case 1: return lambda _: np.array([1., 0.])
#         case 2: return lambda _: np.array([0., 1.])
#     assert False


def get_triangle_grad_fun():
    # returns
    #   dN1/ds, dN1/dt
    #   dN2/ds, dN2/dt
    #   dN3/ds, dN3/dt
    return lambda _: np.array([
        [-1., -1.],
        [ 1.,  0.],
        [ 0.,  1.]
    ])


def triangle_element_grad_var(element_points, elem_grad, i: int, j: int) -> float:
    def integrand(s, t):
        bg = elem_grad(np.array([s, t]))
        J = bg.T @ element_points
        Jinv = inv(J)
        jac = np.abs(det(J))
        return jac * np.dot(Jinv @ bg[i], Jinv @ bg[j])
    # def integrand(s, t):
    #     bg = elem_grad(np.array([s, t]))
    #     J = bg.T @ element_points
    #     U, s, _ = svd(J)
    #     jacobian = np.abs(np.prod(s))
    #     sinv = np.reciprocal(s)
    #     return jacobian * np.dot(sinv * (U.T @ bg[i]), sinv * (U.T @ bg[j]))
    v, _ = dblquad(integrand, 0, 1, 0, lambda s: 1 - s)
    return v


def triangle_element_grad_const(J: np.array, grad_i: np.array, grad_j: np.array) -> float:
    Jinv = inv(J)
    # 0.5 is the integral of 1 with s=0..1, t=0..1-s (area of half a square)
    return 0.5 * np.abs(det(J)) * np.dot(Jinv @ grad_i, Jinv @ grad_j)
    # U, s, _ = svd(J)
    # jacobian = np.abs(np.prod(s))
    # sinv = np.reciprocal(s)
    # d = np.dot(sinv * (U.T @ grad_i), sinv * (U.T @ grad_j))
    # return 0.5 * jacobian * d


def triangle_grad_dot_grad(points: np.array, elements: np.array) -> coo_array:
    order = elements.shape[1]
    elem_grad = get_triangle_grad_fun()

    # pre-allocate np arrays instead?
    row: list[float] = []
    col: list[float] = []
    data: list[float] = []

    for element_idx in elements:
        element_points = points[element_idx, :]

        if callable(elem_grad):
            J = None
        else:
            J = np.dot(elem_grad.T, element_points)

        for i in range(0, order):
            for j in range(0, order):
                # exploit symmetry?
                if J is None:
                    v = triangle_element_grad_var(element_points, elem_grad, i, j)
                else:
                    v = triangle_element_grad_const(J, elem_grad[i], elem_grad[j])
                row.append(element_idx[i])
                col.append(element_idx[j])
                data.append(v)

    return coo_array(
        (data, (row, col)),
        shape=(points.shape[0], points.shape[0]),
        dtype=np.float64
    ).tocsr()


def assemble(mesh: Mesh) -> csr_array:
    points = mesh.points
    dim = points.shape[0]
    A = csr_array((dim, dim), dtype=np.float64)

    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            A += triangle_grad_dot_grad(points, cell_block.data)
        else:
            print(f"Unsupported cell block {cell_block}")

    return A


mesh = rect2d(5, 4, 5, 3)

# mesh.write('rect.vtu')

print("== Mesh")
print(f"points: {mesh.points.shape}")
print("cells")
for cb in mesh.cells:
    print(f"  {cb.type}: {cb.data.shape}")

start = time.time()
A = assemble(mesh)
end = time.time()

print(f"Time: {end - start}")

np.set_printoptions(linewidth=200, precision=4)
print(A.toarray())
