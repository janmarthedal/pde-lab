import time
from numpy import set_printoptions
from meshgen import rect2d
from assemble.grad_dot_grad import assemble_grad_dot_grad

mesh = rect2d(5, 4, 5, 3)

print("== Mesh")
print(f"points: {mesh.points.shape}")
print("cells")
for cb in mesh.cells:
    print(f"  {cb.type}: {cb.data.shape}")

start = time.time()
A = assemble_grad_dot_grad(mesh)
end = time.time()

print(f"Time: {end - start}")

set_printoptions(linewidth=200, precision=4)
print(A.toarray())
