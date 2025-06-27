import time
import numpy as np
from mesh.generate import rect2d
from assemble.matrix import from_bilinear, grad_dot_grad
from assemble.vector import from_linear
# from scipy.spatial.transform import Rotation

mesh = rect2d(5, 4, 5, 3, addz=False)

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

print(f"Time A: {t2 - t1}")
print(f"Time b: {t3 - t2}")

np.set_printoptions(linewidth=200, precision=16)
if A.shape[1] < 0:
    print(A.toarray())
print(b)
