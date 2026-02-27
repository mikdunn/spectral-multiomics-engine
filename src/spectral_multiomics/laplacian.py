from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags, identity
from scipy.sparse.linalg import eigsh


def normalized_laplacian(A: csr_matrix) -> csr_matrix:
    """Compute the symmetric normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}."""
    if not isinstance(A, csr_matrix):
        A = A.tocsr()

    d = np.asarray(A.sum(axis=1)).ravel()
    inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
    D_inv_sqrt = diags(inv_sqrt, offsets=0, format="csr")
    n = A.shape[0]
    L = identity(n, format="csr", dtype=np.float64) - (D_inv_sqrt @ A @ D_inv_sqrt)
    return L


def unnormalized_laplacian(A: csr_matrix) -> csr_matrix:
    """Compute the unnormalized Laplacian: L = D - A."""
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    d = np.asarray(A.sum(axis=1)).ravel()
    D = diags(d, offsets=0, format="csr", dtype=np.float64)
    return D - A


def laplacian_eigenvectors(
    A: csr_matrix,
    k: int,
    *,
    normalized: bool = True,
    include_trivial: bool = False,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the smallest-k Laplacian eigenpairs.

    Args:
        A: adjacency (n x n), assumed symmetric and nonnegative
        k: number of eigenvectors to return
        normalized: whether to use normalized Laplacian
        include_trivial: if False, drops the first (near-zero) eigenvector

    Returns:
        evals: (k,) ascending
        evecs: (n, k) column-orthonormal
    """

    if k <= 0:
        raise ValueError("k must be positive")

    if not isinstance(A, csr_matrix):
        A = A.tocsr()

    n = A.shape[0]
    if A.shape[1] != n:
        raise ValueError("A must be square")

    L = normalized_laplacian(A) if normalized else unnormalized_laplacian(A)

    # If we want to drop the trivial eigenvector (lambda ~ 0), compute one extra.
    k_eff = int(k + (0 if include_trivial else 1))
    if k_eff >= n:
        # Dense fallback (small n): compute all, then slice.
        M = L.toarray()
        w, v = np.linalg.eigh(M)
        w = np.asarray(w, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
    else:
        # Smallest magnitude for PSD matrices.
        w, v = eigsh(L, k=k_eff, which="SM", tol=float(tol), maxiter=maxiter)
        # eigsh doesn't guarantee sorted
        idx = np.argsort(w)
        w = w[idx]
        v = v[:, idx]

    if include_trivial:
        w_out = w[:k]
        v_out = v[:, :k]
    else:
        w_out = w[1 : 1 + k]
        v_out = v[:, 1 : 1 + k]

    return w_out.astype(np.float64), v_out.astype(np.float32)
