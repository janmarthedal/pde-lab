import time
from numpy import set_printoptions
import skfem as fem
from skfem.helpers import dot, grad
from skfem.mesh import Mesh


@fem.BilinearForm
def a(u, v, _):
    print(f"a: {u.shape}, {v.shape}")
    return dot(grad(u), grad(v))


mesh = Mesh.load('rect.vtu')

Vh = fem.Basis(mesh, fem.ElementTriP1())

start = time.time()
A = a.assemble(Vh)
end = time.time()

print(f"Stiffness matrix size: {A.shape[0]}x{A.shape[1]}")
print(f"Time: {end - start}")

if A.shape[1] < 20:
    set_printoptions(linewidth=200, precision=4)
    print(A.toarray())
