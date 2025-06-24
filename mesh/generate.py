from meshio import Mesh
from numpy import linspace, arange, stack, vstack, hstack, zeros, array


def rect2d(x: float, y: float, nx: int, ny: int, addz=False) -> Mesh:
    xs = linspace(0.0, x, nx)
    ys = linspace(0.0, y, ny)
    points = array([(x, y) for y in ys for x in xs])
    if addz:
        points = hstack([points, zeros((points.shape[0], 1))])
    p = arange(0, nx * ny).reshape(ny, nx)[0:-1, 0:-1].ravel()
    triangles = vstack(
        (
            stack((p, p + 1, p + nx + 1), axis=-1),
            stack((p, p + nx + 1, p + nx), axis=-1),
        )
    )
    return Mesh(points, [("triangle", triangles)])
