import time
from numpy import set_printoptions
from mesh.generate import rect2d
from assembly.grad_dot_grad import assemble_grad_dot_grad
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

start = time.time()
A = assemble_grad_dot_grad(mesh)
end = time.time()

print(f"Time: {end - start}")

if A.shape[1] < 20:
    set_printoptions(linewidth=200, precision=4)
    print(A.toarray())
