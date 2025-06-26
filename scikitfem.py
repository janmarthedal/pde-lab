import time
import numpy as np
import skfem as fem
from skfem.helpers import dot, grad
from skfem.mesh import Mesh


@fem.BilinearForm
def a(u, v, _):
    return dot(grad(u), grad(v))


@fem.LinearForm
def l(v, w):
    x, y = w.x  # global coordinates
    f = np.sin(np.pi * x) * np.sin(np.pi * y)
    return f * v


mesh = Mesh.load("rect.vtu")

Vh = fem.Basis(mesh, fem.ElementTriP1())

t1 = time.time()
A = a.assemble(Vh)
t2 = time.time()
b = l.assemble(Vh)
t3 = time.time()

print(f"Stiffness matrix size: {A.shape[0]}x{A.shape[1]}")
print(f"Time A: {t2 - t1}")
print(f"Time b: {t3 - t2}")

np.set_printoptions(linewidth=200, precision=4)

if A.shape[1] < 0:
    print(A.toarray())
print(b)
