# https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-1-poisson-equation-with-unit-load

import numpy as np
from matplotlib import pyplot as plt
from pardeqsi.mesh.generate import rect2d
from pardeqsi.mesh.boundary import mesh_boundary
from pardeqsi.assemble.matrix import from_bilinear, grad_dot_grad
from pardeqsi.assemble.vector import from_linear
from pardeqsi.solver import solve


def linear_fn(vals: np.ndarray, _coords: np.ndarray):
    return vals


mesh = rect2d(1.0, 1.0, 65, 65, addz=False)
triangles = mesh.cells[0].data

print(f"points: {mesh.points.shape}")
print(f"triangles: {triangles.shape[0]}")

A = from_bilinear(mesh, grad_dot_grad)
b = from_linear(mesh, linear_fn, norder=2)

bmesh = mesh_boundary(mesh)
idx_bdy = bmesh.point_data["point_idx"]
x = solve(A, b, idx_bdy, np.zeros_like(idx_bdy))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect(1.0)
im = ax.tripcolor(
    mesh.points[:, 0],
    mesh.points[:, 1],
    x,
    triangles=triangles,
    cmap=plt.cm.jet,
    shading="gouraud",
)
plt.colorbar(im, ax=ax)
plt.show()
