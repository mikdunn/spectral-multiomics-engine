from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np

from .binning import GenomicBins
from .reporting import ensure_dir, write_json


def make_synthetic_contact_map(
    n_bins: int,
    *,
    seed: int = 0,
    bandwidth: int = 32,
    scale: float = 12.0,
    community_boost: float = 2.0,
) -> dict[str, np.ndarray]:
    """Create a small, sparse, symmetric contact map with distance decay + block structure."""
    rng = np.random.default_rng(int(seed))
    n = int(n_bins)
    bw = int(bandwidth)

    rows = []
    cols = []
    data = []

    # Two communities
    half = n // 2

    for i in range(n):
        j0 = max(0, i - bw)
        j1 = min(n - 1, i + bw)
        for j in range(j0, j1 + 1):
            dist = abs(i - j)
            base = np.exp(-dist / float(scale))
            # community boost
            same_block = (i < half and j < half) or (i >= half and j >= half)
            if same_block:
                base *= float(community_boost)
            # add small noise, keep nonnegative
            w = base + 0.02 * rng.standard_normal()
            if w <= 0:
                continue
            rows.append(i)
            cols.append(j)
            data.append(float(w))

    row = np.asarray(rows, dtype=np.int64)
    col = np.asarray(cols, dtype=np.int64)
    val = np.asarray(data, dtype=np.float64)

    # symmetrize by mirroring (and then we'll sum duplicates when loading)
    row2 = np.concatenate([row, col])
    col2 = np.concatenate([col, row])
    val2 = np.concatenate([val, val])

    return {
        "row": row2,
        "col": col2,
        "data": val2,
        "shape": np.asarray([n, n], dtype=np.int64),
    }


def make_synthetic_signal(bins: GenomicBins, *, seed: int = 0) -> np.ndarray:
    """Create a per-bin signal with smooth structure (for end-to-end sanity checks)."""
    rng = np.random.default_rng(int(seed))
    x = np.linspace(0, 1, bins.n_bins, endpoint=False, dtype=np.float64)
    y = (
        1.2 * np.sin(2 * np.pi * x)
        + 0.6 * np.sin(6 * np.pi * x + 0.3)
        + 0.25 * rng.standard_normal(bins.n_bins)
    )
    # add a step change between communities
    y[bins.n_bins // 2 :] += 0.8
    return y.astype(np.float32)


def write_bedgraph_from_binned_signal(
    y: np.ndarray,
    bins: GenomicBins,
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    starts = bins.start + bins.binsize * np.arange(bins.n_bins, dtype=np.int64)
    ends = starts + bins.binsize

    # One row per bin interval. This makes binning round-trip exact.
    lines = []
    for s, e, v in zip(starts, ends, y):
        lines.append(f"{bins.chrom}\t{int(s)}\t{int(e)}\t{float(v):.6f}\n")

    out_path.write_text("".join(lines))


def synth_dataset(
    out_dir: str | Path,
    *,
    n_bins: int = 256,
    binsize: int = 100_000,
    chrom: str = "chr1",
    seed: int = 0,
) -> dict[str, Path]:
    out_dir = ensure_dir(out_dir)

    bins = GenomicBins(chrom=chrom, start=0, binsize=int(binsize), n_bins=int(n_bins))

    contact = make_synthetic_contact_map(n_bins=bins.n_bins, seed=seed)
    contact_path = out_dir / "contact_map.npz"
    np.savez_compressed(contact_path, **contact)

    y = make_synthetic_signal(bins, seed=seed)
    signal_path = out_dir / "signal.bedgraph.tsv"
    write_bedgraph_from_binned_signal(y, bins, signal_path)

    meta = {
        "chrom": chrom,
        "binsize": int(binsize),
        "n_bins": int(n_bins),
        "seed": int(seed),
        "contact_format": ".npz COO (row,col,data,shape)",
        "signal_format": "bedGraph-like TSV (chrom,start,end,value)",
    }
    write_json(meta, out_dir / "meta.json")

    return {
        "contact": contact_path,
        "signal": signal_path,
        "meta": out_dir / "meta.json",
    }
