from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .binning import GenomicBins, bin_bedgraph, bin_bigwig
from .contact_map import load_contact_map, load_contact_map_cooler
from .laplacian import laplacian_eigenvectors
from .projection import project_signal
from .reporting import ensure_dir, write_json


@dataclass(frozen=True)
class PipelineOutputs:
    out_dir: Path
    evals_path: Path
    evecs_path: Path
    signal_binned_path: Path
    projection_path: Path
    meta_path: Path


def run_pipeline(
    *,
    contact: str | Path,
    signal: str | Path,
    out_dir: str | Path,
    binsize: int,
    chrom: str = "chr1",
    k: int = 16,
    normalized_laplacian: bool = True,
    include_trivial: bool = False,
    agg: str = "mean",
) -> PipelineOutputs:
    out_dir = ensure_dir(out_dir)

    contact = Path(contact)
    if contact.suffix.lower() in {".cool", ".mcool"}:
        A = load_contact_map_cooler(contact, chrom=chrom)
    else:
        A = load_contact_map(contact)
    n_bins = int(A.shape[0])

    bins = GenomicBins(chrom=chrom, start=0, binsize=int(binsize), n_bins=n_bins)

    evals, evecs = laplacian_eigenvectors(
        A,
        k=int(k),
        normalized=bool(normalized_laplacian),
        include_trivial=bool(include_trivial),
    )

    signal = Path(signal)
    if signal.suffix.lower() in {".bw", ".bigwig", ".bigwig", ".bigwიგ"}:
        # Note: keep suffix check conservative; primary patterns are .bw and .bigWig
        y = bin_bigwig(signal, bins, agg=agg)
    else:
        y = bin_bedgraph(signal, bins, agg=agg)
    coeffs = project_signal(evecs, y)

    evals_path = out_dir / "evals.npy"
    evecs_path = out_dir / "evecs.npy"
    signal_binned_path = out_dir / "signal_binned.npy"
    projection_path = out_dir / "projection.npy"
    meta_path = out_dir / "meta.json"

    np.save(evals_path, evals)
    np.save(evecs_path, evecs)
    np.save(signal_binned_path, y)
    np.save(projection_path, coeffs)

    meta = {
        "contact": str(contact),
        "signal": str(signal),
        "chrom": chrom,
        "binsize": int(binsize),
        "n_bins": int(n_bins),
        "k": int(k),
        "normalized_laplacian": bool(normalized_laplacian),
        "include_trivial": bool(include_trivial),
        "agg": agg,
    }
    write_json(meta, meta_path)

    return PipelineOutputs(
        out_dir=Path(out_dir),
        evals_path=evals_path,
        evecs_path=evecs_path,
        signal_binned_path=signal_binned_path,
        projection_path=projection_path,
        meta_path=meta_path,
    )
