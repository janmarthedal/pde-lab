import numpy as np
from meshio import Mesh


_BOUNDARY_SELECTORS = {
    "line": ("vertex", [[0], [1]]),
    "triangle": ("line", [[0, 1], [1, 2], [2, 0]]),
}


def _mesh_boundary(mesh: Mesh) -> Mesh:
    new_cells = []
    for cell_block in mesh.cells:
        try:
            bdy_type, selectors = _BOUNDARY_SELECTORS[cell_block.type]
        except KeyError:
            raise RuntimeError(f"Unsupported cell block {cell_block}")
        segments = np.vstack([cell_block.data[:, col_idxs] for col_idxs in selectors])
        segments_normalized = segments.copy()
        segments_normalized.sort(axis=1)
        segment_keys = segments_normalized.view(
            dtype=np.dtype(
                [("", segments_normalized.dtype)] * segments_normalized.shape[1]
            )
        ).ravel()
        _, idx, counts = np.unique(segment_keys, return_index=True, return_counts=True)
        new_cells.append((bdy_type, segments[idx[counts == 1]]))
    return Mesh(mesh.points, new_cells)


def _mesh_prune_points(mesh: Mesh) -> Mesh:
    # `point_set` maps new index to old index
    point_set = np.unique(
        np.concatenate([cell_block.data.ravel() for cell_block in mesh.cells]),
        sorted=True,
    )
    # `dtype` to use for indexing
    idx_dtype = point_set.dtype
    # `rev` maps old index to new index
    rev = np.zeros((len(mesh.points),), dtype=idx_dtype)
    rev[point_set] = np.arange(len(point_set))
    points = mesh.points[point_set]
    cells = [(cell_block.type, rev[cell_block.data]) for cell_block in mesh.cells]
    point_data = {key: data[point_set] for key, data in mesh.point_data.items()}
    return Mesh(points, cells, point_data=point_data)


def mesh_boundary(mesh: Mesh) -> Mesh:
    bdy_mesh = _mesh_boundary(mesh)
    bdy_mesh.point_data["point_idx"] = np.arange(len(mesh.points), dtype=np.uint32)
    return _mesh_prune_points(bdy_mesh)
