from scipy import sparse
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils import sparsefuncs_fast


def axis_norms(X: sparse.csr_matrix, norm: str = "l1", axis: int = 1) -> np.ndarray:
    if norm == "l1":
        return np.asarray(X.sum(axis=axis)).reshape(-1)
    return np.sqrt(np.asarray(X.power(2).sum(axis=axis)).reshape(-1))


def row_normalize_l1(X: sparse.csr_matrix, inplace: bool = True):
    if inplace:
        sparsefuncs_fast._inplace_csr_row_normalize_l1(
            X.data, X.shape, X.indices, X.indptr
        )
        return X
    return normalize(X, norm="l1", copy=True)


def anisotropic_laplacian(
    X: sparse.csr_matrix, alpha: float = 1.0
) -> sparse.csr_matrix:
    D_m = sparse.diags(axis_norms(X)).power(-alpha)
    D_n = sparse.diags(axis_norms(X, axis=0)).power(-alpha)
    return D_m @ X @ D_n
