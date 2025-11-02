# https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-15-one-dimensional-poisson-equation

import numpy as np
from matplotlib import pyplot as plt
from pardeqsi.mesh.generate import line
from pardeqsi.mesh.boundary import mesh_boundary
from pardeqsi.assemble.matrix import from_bilinear, grad_dot_grad
from pardeqsi.assemble.vector import from_linear
from pardeqsi.solver import solve


def unit_load(vals: np.ndarray, _coords: np.ndarray):
    return vals


mesh = line(np.linspace(0, 1, 10))

A = from_bilinear(mesh, grad_dot_grad)
b = from_linear(mesh, unit_load, norder=1)

bmesh = mesh_boundary(mesh)
idx_bdy = bmesh.point_data["point_idx"]
x = solve(A, b, idx_bdy, np.zeros_like(idx_bdy))

plt.plot(mesh.points[:, 0], x)
# plt.xlabel('x')
# plt.ylabel('u(x)')
# plt.title('Solution of the Poisson equation')
plt.show()
