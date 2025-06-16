from collections.abc import Callable
import numpy as np
from meshio import Mesh
from meshgen import rect2d
from scipy.integrate import dblquad
from scipy.linalg import det, inv, solve
from scipy.sparse import coo_array, csr_array


def get_triangle_fun(i: int) -> Callable[[np.array], np.array]:
    match i:
        case 0: return lambda p: 1 - p[0] - p[1]
        case 1: return lambda p: p[0]
        case 2: return lambda p: p[1]
    assert False


def get_triangle_grad_fun(i: int) -> Callable[[np.array], np.array]:
    match i:
        case 0: return lambda _: np.array([-1., -1.])
        case 1: return lambda _: np.array([1., 0.])
        case 2: return lambda _: np.array([0., 1.])
    assert False


def triangle_integrate_2d(points: np.array, i: int, j: int) -> float:
    assert points.shape == (3, 2)

    # mapping local coordinates R^2 -> R^2, s=0..1, t=0..1-s

    # The Jacobian matrix
    J = points[1:3] - points[0]
    # The Jacobian
    jac = np.abs(det(J))

    f1 = get_triangle_fun(i)
    f2 = get_triangle_fun(j)
    f = lambda s, t: f1([s, t]) * f2([s, t])
    v, _ = dblquad(f, 0, 1, 0, lambda s: 1 - s)

    return jac * v


def triangle_integrate_grad_2d(points: np.array, i: int, j: int) -> float:
    assert points.shape == (3, 2)

    # mapping local coordinates R^2 -> R^2, s=0..1, t=0..1-s

    # The Jacobian matrix
    J = points[1:3] - points[0]
    Jinv = inv(J)
    # The Jacobian
    jac = np.abs(det(J))

    f1 = get_triangle_grad_fun(i)
    f2 = get_triangle_grad_fun(j)
    f = lambda s, t: np.dot(np.dot(Jinv, f1([s, t])), np.dot(Jinv, f2([s, t])))
    v, _ = dblquad(f, 0, 1, 0, lambda s: 1 - s)

    return jac * v


def assemble(mesh: Mesh) -> csr_array:
    assert mesh.points.shape[1] == 2
    point_count = mesh.points.shape[0]
    print(f"point_count {point_count}")

    # pre-allocate np arrays instead?
    row: list[float] = []
    col: list[float] = []
    data: list[float] = []

    for cell_block in mesh.cells:
        if cell_block.type == 'triangle':
            for element_idx in cell_block.data:
                element_points = mesh.points[element_idx, :]
                for i in range(0, 3):
                    for j in range(0, 3):
                        # exploit symmetry?
                        v = triangle_integrate_grad_2d(element_points, i, j)
                        row.append(element_idx[i])
                        col.append(element_idx[j])
                        data.append(v)
        else:
            print(f"Unsupported cell block {cell_block}")

    A = coo_array((data, (row, col)), shape=(point_count, point_count), dtype=np.float64)
    return A.tocsr()


mesh = rect2d(5, 4, 5, 3)

# mesh.write('rect.vtu')

print("== Mesh")
print(f"points: {mesh.points.shape}")
print("cells")
for cb in mesh.cells:
    print(f"  {cb.type}: {cb.data.shape}")

A = assemble(mesh)

np.set_printoptions(linewidth=200)
print(A.toarray())
