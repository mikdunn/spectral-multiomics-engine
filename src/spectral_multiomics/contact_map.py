from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix


def _load_edge_list_tsv(path: Path) -> csr_matrix:
    df = pd.read_csv(path, sep="\t", header=None, names=["i", "j", "w"])
    if df.shape[1] < 2:
        raise ValueError("Edge list must have at least two columns: i, j (optionally w)")
    if "w" not in df.columns:
        df["w"] = 1.0

    i = df["i"].to_numpy(dtype=np.int64)
    j = df["j"].to_numpy(dtype=np.int64)
    w = df["w"].to_numpy(dtype=np.float64)

    if i.min(initial=0) < 0 or j.min(initial=0) < 0:
        raise ValueError("Edge indices must be non-negative")

    n = int(max(i.max(initial=0), j.max(initial=0)) + 1)

    # Symmetrize (undirected contact map)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.concatenate([w, w])

    A = coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()
    A.sum_duplicates()
    return A


def load_contact_map(path: str | Path) -> csr_matrix:
    """Load a contact map adjacency matrix.

    Supported:
      - .npy dense square matrix
      - .npz sparse COO with keys: row, col, data, shape
      - .tsv edge list with columns: i, j, weight (0-based indices)

    Returns:
        A: csr_matrix (n x n), symmetric (for .tsv and .npz as provided)
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    suf = path.suffix.lower()
    if suf == ".npy":
        M = np.load(path)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"Expected square 2D matrix in {path}")
        return csr_matrix(M.astype(np.float64, copy=False))

    if suf in {".cool", ".mcool"}:
        raise ValueError(
            "Cooler formats (.cool/.mcool) require specifying a chromosome/region. "
            "Call load_contact_map_cooler(path, chrom=...) or use the CLI demo command."
        )

    if suf == ".npz":
        z = np.load(path, allow_pickle=False)
        for k in ("row", "col", "data", "shape"):
            if k not in z:
                raise ValueError(f"Missing key {k!r} in {path} (.npz contact map)")
        row = z["row"].astype(np.int64, copy=False)
        col = z["col"].astype(np.int64, copy=False)
        data = z["data"].astype(np.float64, copy=False)
        shape = tuple(int(x) for x in z["shape"])
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError(f"Contact map shape must be square; got {shape}")
        A = coo_matrix((data, (row, col)), shape=shape, dtype=np.float64).tocsr()
        A.sum_duplicates()
        return A

    if suf in {".tsv", ".txt", ".csv"}:
        return _load_edge_list_tsv(path)

    raise ValueError(f"Unsupported contact map format: {path}")


def load_contact_map_cooler(path: str | Path, *, chrom: str) -> csr_matrix:
    """Load a per-chromosome contact map from a Cooler file (.cool/.mcool).

    Notes:
      - This is an *optional* feature: it requires the third-party `cooler` package.
      - For now we keep the MVP "single chromosome" assumption by requiring `chrom`.

    Args:
        path: Path to .cool or .mcool
        chrom: Chromosome/contig name to fetch (e.g. 'chr1', 'chrI')

    Returns:
        A: csr_matrix (n_bins x n_bins), symmetric
    """

    try:
        import cooler  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Loading .cool/.mcool contact maps requires the optional dependency 'cooler'. "
            "Install with: pip install 'spectral-multiomics-engine[hic]' (or pip install cooler)."
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    if path.suffix.lower() not in {".cool", ".mcool"}:
        raise ValueError(f"Expected a .cool/.mcool file; got: {path}")

    c = cooler.Cooler(str(path))
    chrom = str(chrom)

    # Fetch sparse matrix for a single region.
    M = c.matrix(balance=False, sparse=True).fetch(chrom)
    A = M.tocsr().astype(np.float64, copy=False)

    # Enforce symmetry (some coolers may store only upper triangle).
    d = A.diagonal()
    A2 = (A + A.T).tocsr()
    A2.setdiag(d)
    A2.eliminate_zeros()

    return A2
