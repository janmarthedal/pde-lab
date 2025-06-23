from meshio import Mesh
import numpy as np

def rect2d(x: float, y: float, nx: float, ny: float) -> Mesh:
    xs = np.linspace(0.0, x, nx)
    ys = np.linspace(0.0, y, ny)
    points = [(x, y) for y in ys for x in xs]
    p = np.arange(0, nx * ny).reshape(ny, nx)[0:-1, 0:-1].ravel()
    triangles = np.vstack((
        np.stack((p, p + 1, p + nx + 1), axis=-1),
        np.stack((p, p + nx + 1, p + nx), axis=-1)
    ))
    return Mesh(points, [('triangle', triangles)])
