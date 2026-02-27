import numpy as np
from scipy.sparse import csr_matrix

from spectral_multiomics.laplacian import laplacian_eigenvectors


def test_laplacian_eigenvectors_shapes():
    # Simple path graph adjacency
    n = 20
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    A = csr_matrix(A)

    evals, evecs = laplacian_eigenvectors(A, k=3, normalized=True, include_trivial=False)
    assert evals.shape == (3,)
    assert evecs.shape == (n, 3)

    # Orthonormal columns (approximately)
    G = evecs.T @ evecs
    assert np.allclose(G, np.eye(3), atol=1e-3)
