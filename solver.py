import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import spsolve


def solve(
    A: spmatrix, b: np.ndarray, idx_bdy: np.ndarray, x_bdy: np.ndarray
) -> np.ndarray:
    idx_free = np.setdiff1d(
        np.arange(len(b), dtype=np.uint32), idx_bdy, assume_unique=True
    )
    A_free = A[idx_free][:, idx_free]
    b_free = b[idx_free] - A[idx_free][:, idx_bdy] @ x_bdy
    x_free = spsolve(A_free, b_free)
    x = np.empty((len(b),), dtype=np.float64)
    x[idx_free] = x_free
    x[idx_bdy] = x_bdy
    return x
