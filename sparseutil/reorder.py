from typing import Tuple

import numpy as np
from scipy import sparse
from sklearn.utils import sparsefuncs


def _apply_perm(
    X: sparse.csr_matrix, p_row: np.ndarray = None, p_col: np.ndarray = None
) -> sparse.csr_matrix:
    if p_row is not None:
        for i, j in enumerate(p_row):
            if i >= j:
                continue
            sparsefuncs.inplace_swap_row(X, i, j)
    if p_col is not None:
        for i, j in enumerate(p_col):
            if i >= j:
                continue
            sparsefuncs.inplace_swap_column(X, i, j)
    return X


def _perm_first_axis(X: sparse.csr_matrix) -> np.ndarray:
    nnz_per_row = X.indptr[1:] - X.indptr[:-1]
    idx_full, = np.where(nnz_per_row)
    idx_empty, = np.where(nnz_per_row == 0)
    p_row = np.arange(X.shape[0])
    for eidx, fidx in zip(idx_empty, idx_full[::-1]):
        if eidx >= fidx:
            break
        p_row[eidx], p_row[fidx] = p_row[fidx], p_row[eidx]

    return p_row


def _reorder_empty(
    X: sparse.csr_matrix, rows: bool = True, cols: bool = True, copy: bool = True
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    if copy:
        Y = X.copy()
    else:
        Y = X
    p_row = None
    p_col = None
    if rows:
        p_row = _perm_first_axis(Y)
    if cols:
        p_col = _perm_first_axis(Y.tocsc())

    Y = _apply_perm(X, p_row=p_row, p_col=p_col)

    return Y, p_row, p_col


def _trim_empty(
    X: sparse.csr_matrix, rows: bool = True, cols: bool = True
) -> sparse.csr_matrix:
    if cols:
        X = X[:, : np.max(X.indices) + 1]
    if rows:
        X = X[: np.where(X.getnnz(axis=1))[0][-1] + 1, :]
    return X


def reorder_trim(
    X: sparse.csr_matrix, rows: bool = True, cols: bool = True, copy: bool = True
) -> Tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    Y, p_row, p_col = _reorder_empty(X, rows=rows, cols=cols, copy=copy)
    return _trim_empty(Y, rows=rows, cols=cols), p_row, p_col


def invert_reorder_trim(
    X: sparse.csr_matrix, p_row: np.ndarray, p_col: np.ndarray
) -> sparse.csr_matrix:
    X.resize(p_row.shape[0], p_col.shape[0])
    return _apply_perm(X, p_row=p_row, p_col=p_col)
