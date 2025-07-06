import meshio
from mesh.mesh import Mesh


def to_meshio(mesh: Mesh) -> meshio.Mesh:
    return meshio.Mesh(
        mesh.points,
        [(cb.type, cb.data) for cb in mesh.cells],
        point_data=mesh.point_data,
    )
