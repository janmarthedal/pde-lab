import numpy as np
from .mesh import Mesh


def rect2d(x: float, y: float, nx: int, ny: int, addz=False) -> Mesh:
    xs = np.linspace(0.0, x, nx)
    ys = np.linspace(0.0, y, ny)
    points = np.array([(x, y) for y in ys for x in xs])
    if addz:
        points = np.hstack([points, np.zeros((points.shape[0], 1))])
    p = np.arange(0, nx * ny, dtype=np.uint32).reshape(ny, nx)[0:-1, 0:-1].ravel()
    triangles = np.vstack(
        (
            np.stack((p, p + 1, p + nx + 1), axis=-1),
            np.stack((p, p + nx + 1, p + nx), axis=-1),
        )
    )
    return Mesh(points, [("triangle", triangles)])
