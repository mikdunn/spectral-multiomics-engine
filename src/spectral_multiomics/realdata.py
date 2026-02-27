ENCODE_GM12878_ATAC_GRCH38_BIGWIG_URL = "https://www.encodeproject.org/files/ENCFF603BJO/@@download/ENCFF603BJO.bigWig"

def download_encode_bigwig_gm12878_grch38(out_path: str | Path, overwrite: bool = False) -> Path:
    """Download GM12878 ATAC-seq bigWig (GRCh38) from ENCODE."""
    return download_url(ENCODE_GM12878_ATAC_GRCH38_BIGWIG_URL, out_path, overwrite=overwrite)
from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contact_map import load_contact_map_cooler
from .reporting import ensure_dir, write_json


DEFAULT_COOLER_YEAST_URL = (
    "https://github.com/open2c/cooler/raw/refs/heads/master/tests/data/yeast.10kb.cool"
)


@dataclass(frozen=True)
class CoolerDemoPaths:
    cool_path: Path
    signal_path: Path
    meta_path: Path
    chrom: str
    binsize: int


def download_url(url: str, out_path: str | Path, *, overwrite: bool = False) -> Path:
    """Download a URL to disk.

    Uses stdlib only (no requests dependency).
    """

    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if out_path.exists() and not overwrite:
        return out_path

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)

    tmp.replace(out_path)
    return out_path


def _require_cooler():
    try:
        import cooler  # type: ignore

        return cooler
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "This feature requires the optional dependency 'cooler'. "
            "Install with: pip install 'spectral-multiomics-engine[hic]' (or pip install cooler)."
        ) from e


def write_bedgraph_from_vector(
    chrom: str,
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    out_path: str | Path,
) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)

    if not (len(starts) == len(ends) == len(values)):
        raise ValueError("starts/ends/values must have the same length")

    lines: list[str] = []
    for s, e, v in zip(starts, ends, values):
        lines.append(f"{chrom}\t{int(s)}\t{int(e)}\t{float(v)}\n")

    out_path.write_text("".join(lines))
    return out_path


def prepare_cooler_demo_dataset(
    *,
    out_dir: str | Path,
    url: str = DEFAULT_COOLER_YEAST_URL,
    chrom: str | None = None,
    overwrite: bool = False,
) -> CoolerDemoPaths:
    """Download a small real Hi-C contact map and create a matching 1D signal.

    The "signal" we generate is the per-bin Hi-C coverage (row sums) for the chosen chromosome.
    It's not an ATAC/RNA track, but it is real experimental data and exercises the full pipeline.

    Args:
        out_dir: output directory under which we create `contact.cool`, `signal.bedgraph.tsv`, `meta.json`
        url: URL to a .cool file
        chrom: chromosome/contig to extract (defaults to first contig in the .cool)
        overwrite: whether to re-download / regenerate
    """

    cooler = _require_cooler()

    out_dir = ensure_dir(out_dir)
    cool_path = out_dir / "contact.cool"
    download_url(url, cool_path, overwrite=overwrite)

    c = cooler.Cooler(str(cool_path))
    if chrom is None:
        # Prefer the largest contig so the demo is non-trivial.
        try:
            chrom = str(c.chromsizes.sort_values(ascending=False).index[0])
        except Exception:
            chrom = str(c.chromnames[0])

    bins = c.bins().fetch(chrom)
    if {"start", "end"}.issubset(bins.columns):
        starts = bins["start"].to_numpy(dtype=np.int64)
        ends = bins["end"].to_numpy(dtype=np.int64)
    else:
        # Fallback: assume fixed-width bins.
        bs = int(c.binsize)
        n = int(c.extent(chrom)[1] - c.extent(chrom)[0])
        starts = bs * np.arange(n, dtype=np.int64)
        ends = starts + bs

    A = load_contact_map_cooler(cool_path, chrom=str(chrom))
    coverage = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)

    signal_path = out_dir / f"signal.{chrom}.coverage.bedgraph.tsv"
    write_bedgraph_from_vector(str(chrom), starts, ends, coverage, signal_path)

    meta = {
        "source_url": url,
        "cool_path": str(cool_path),
        "chrom": str(chrom),
        "binsize": int(c.binsize) if c.binsize is not None else None,
        "n_bins": int(A.shape[0]),
        "signal": "Hi-C marginal coverage (row sums)",
    }
    meta_path = out_dir / "meta.json"
    write_json(meta, meta_path)

    binsize = int(c.binsize) if c.binsize is not None else int(ends[0] - starts[0])

    return CoolerDemoPaths(
        cool_path=cool_path,
        signal_path=signal_path,
        meta_path=meta_path,
        chrom=str(chrom),
        binsize=binsize,
    )
