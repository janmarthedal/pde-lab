import time
import numpy as np
from pardeqsi.mesh.generate import rect2d
from pardeqsi.mesh.boundary import mesh_boundary
from pardeqsi.assemble.matrix import from_bilinear, grad_dot_grad
from pardeqsi.assemble.vector import from_linear
from pardeqsi.solver import solve

mesh = rect2d(1.0, 1.0, 90, 90, addz=False)

# r = Rotation.from_euler('zyx', [90, 45, 30], degrees=True)
# mesh.points = r.apply(mesh.points)

# mesh.write('rect.vtu')

print("== Mesh")
print(f"points: {mesh.points.shape}")
print("cells")
for cb in mesh.cells:
    print(f"  {cb.type}: {cb.data.shape}")

# import cProfile
# cProfile.run('assemble_grad_dot_grad(mesh)')


def linear_fn(vals: np.ndarray, coords: np.ndarray):
    return np.prod(np.sin(np.pi * coords), axis=0) * vals


t1 = time.time()
A = from_bilinear(mesh, grad_dot_grad)
t2 = time.time()
b = from_linear(mesh, linear_fn, norder=2)
t3 = time.time()

bmesh = mesh_boundary(mesh)
idx_bdy = bmesh.point_data["point_idx"]
x = solve(A, b, idx_bdy, np.zeros_like(idx_bdy))
print(x)
# mesh.point_data["values"] = x
# mesh.write("solution.vtu")

print(f"Time A: {t2 - t1}")
print(f"Time b: {t3 - t2}")

np.set_printoptions(linewidth=200, precision=16)
if A.shape[1] < 0:
    print(A.toarray())
    print(b)
