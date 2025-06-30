import numpy as np
from meshio import Mesh


BOUNDARY_SELECTORS = {
    "line": ("vertex", [[0], [1]]),
    "triangle": ("line", [[0, 1], [1, 2], [2, 0]]),
}


def mesh_boundary(mesh: Mesh) -> Mesh:
    new_cells = []
    for cell_block in mesh.cells:
        try:
            bdy_type, selectors = BOUNDARY_SELECTORS[cell_block.type]
        except KeyError:
            raise RuntimeError(f"Unsupported cell block {cell_block}")
        segments = np.vstack([cell_block.data[:, col_idxs] for col_idxs in selectors])
        segments_normalized = segments.copy()
        segments_normalized.sort(axis=1)
        segment_keys = segments_normalized.view(
            dtype=np.dtype(
                [("", segments_normalized.dtype)] * segments_normalized.shape[1]
            )
        ).reshape(-1)
        _, idx, counts = np.unique(segment_keys, return_index=True, return_counts=True)
        new_cells.append((bdy_type, segments[idx[counts == 1]]))
    return Mesh(mesh.points, new_cells)


def mesh_prune_points(mesh: Mesh) -> Mesh:
    # `point_set` maps new index to old index
    point_set = np.array([], dtype=np.uint32)
    for cell_block in mesh.cells:
        point_set = np.unique(np.hstack([point_set, cell_block.data.reshape(-1)]))
    point_set.sort()
    # `dtype` to use for indexing
    idx_dtype = point_set.dtype
    # `rev` maps old index to new index
    rev = np.zeros((len(mesh.points),), dtype=idx_dtype)
    rev[point_set] = np.arange(len(point_set))
    if "point_idx" in mesh.point_data:
        cur_point_idx = mesh.point_data["point_idx"]
    else:
        cur_point_idx = np.arange(len(mesh.points), dtype=idx_dtype)
    points = mesh.points[point_set]
    point_idx = cur_point_idx[point_set]
    cells = [(cell_block.type, rev[cell_block.data]) for cell_block in mesh.cells]
    return Mesh(points, cells, point_data={"point_idx": point_idx})
