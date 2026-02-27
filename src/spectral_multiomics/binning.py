from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GenomicBins:
    """Simple fixed-width bins for a single chromosome.

    MVP simplification: single chrom per run.
    """

    chrom: str
    start: int
    binsize: int
    n_bins: int

    @property
    def end(self) -> int:
        return self.start + self.binsize * self.n_bins

    def bin_index(self, pos: int) -> int:
        return int((pos - self.start) // self.binsize)

    def to_dataframe(self) -> pd.DataFrame:
        starts = self.start + self.binsize * np.arange(self.n_bins, dtype=np.int64)
        ends = starts + self.binsize
        return pd.DataFrame(
            {
                "chrom": self.chrom,
                "start": starts,
                "end": ends,
                "bin": np.arange(self.n_bins, dtype=np.int64),
            }
        )


def _read_bedgraph(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "value"],
        dtype={"chrom": str, "start": np.int64, "end": np.int64, "value": np.float64},
    )
    if df.empty:
        raise ValueError(f"Signal file is empty: {path}")
    if (df["end"] <= df["start"]).any():
        bad = df.index[df["end"] <= df["start"]][0]
        raise ValueError(f"Invalid interval with end<=start at row {bad}")
    return df


def bin_bedgraph(
    path: str | Path,
    bins: GenomicBins,
    *,
    agg: str = "mean",
    fill: float = 0.0,
) -> np.ndarray:
    """Bin a bedGraph-like TSV into fixed-width genomic bins.

    The bedGraph is expected to have columns: chrom, start, end, value.

    Args:
        agg: "mean" (length-weighted mean per bin) or "sum" (length-weighted sum per bin)
        fill: value used for bins with no coverage

    Returns:
        y: shape (n_bins,)
    """

    df = _read_bedgraph(path)
    df = df[df["chrom"] == bins.chrom]
    if df.empty:
        raise ValueError(f"No rows for chrom={bins.chrom} in {path}")

    n = int(bins.n_bins)
    total = np.zeros(n, dtype=np.float64)
    covered = np.zeros(n, dtype=np.float64)  # bp covered in each bin

    bin0_start = int(bins.start)
    bin_size = int(bins.binsize)
    bin_end = bins.end

    for s, e, v in zip(df["start"].to_numpy(), df["end"].to_numpy(), df["value"].to_numpy()):
        s = int(s)
        e = int(e)
        if e <= bin0_start or s >= bin_end:
            continue
        # clip to bin span
        s = max(s, bin0_start)
        e = min(e, bin_end)

        b0 = (s - bin0_start) // bin_size
        b1 = (e - 1 - bin0_start) // bin_size
        b0 = int(max(b0, 0))
        b1 = int(min(b1, n - 1))

        for b in range(b0, b1 + 1):
            bs = bin0_start + b * bin_size
            be = bs + bin_size
            ov = max(0, min(e, be) - max(s, bs))
            if ov:
                total[b] += float(v) * ov
                covered[b] += ov

    if agg.lower() == "sum":
        y = total
    elif agg.lower() == "mean":
        y = np.full(n, float(fill), dtype=np.float64)
        m = covered > 0
        y[m] = total[m] / covered[m]
    else:
        raise ValueError(f"Unknown agg={agg!r}; expected 'mean' or 'sum'")

    return y.astype(np.float32)


def bin_bigwig(
    path: str | Path,
    bins: GenomicBins,
    *,
    agg: str = "mean",
    fill: float = 0.0,
) -> np.ndarray:
    """Bin a bigWig track into fixed-width genomic bins.

    This is intended for ENCODE/UCSC-style signal tracks.

    Args:
        path: .bw/.bigWig file
        bins: GenomicBins defining chromosome span and bin width
        agg: "mean" or "sum" (delegated to bigWig summary stats)
        fill: used when a bin has no data (bigWig stats returns None)

    Returns:
        y: shape (n_bins,)
    """

    try:
        import pyBigWig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Reading bigWig files requires the optional dependency 'pyBigWig'. "
            "Install with: pip install 'spectral-multiomics-engine[encode]' (or pip install pyBigWig)."
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    stat_type = agg.lower()
    if stat_type not in {"mean", "sum"}:
        raise ValueError(f"Unknown agg={agg!r}; expected 'mean' or 'sum'")

    bw = pyBigWig.open(str(path))
    try:
        chroms = bw.chroms()
        if bins.chrom not in chroms:
            # Provide a more helpful error than pyBigWig's KeyError.
            preview = ", ".join(list(chroms.keys())[:10])
            raise ValueError(
                f"Chrom {bins.chrom!r} not found in bigWig {path}. "
                f"First contigs: {preview}{'...' if len(chroms) > 10 else ''}"
            )

        n = int(bins.n_bins)
        y = np.full(n, float(fill), dtype=np.float64)
        starts = bins.start + bins.binsize * np.arange(n, dtype=np.int64)
        ends = starts + bins.binsize

        # bigWig summary stats are computed over the interval; None means no data.
        for i, (s, e) in enumerate(zip(starts, ends)):
            v = bw.stats(bins.chrom, int(s), int(e), type=stat_type, exact=True)[0]
            if v is None:
                continue
            y[i] = float(v)

        return y.astype(np.float32)
    finally:
        try:
            bw.close()
        except Exception:
            pass


def bin_bigwig(
    path: str | Path,
    bins: GenomicBins,
    *,
    agg: str = "mean",
    fill: float = 0.0,
) -> np.ndarray:
    """Bin a bigWig track into fixed-width genomic bins.

    This is intended for real signal tracks (e.g. ENCODE ATAC-seq / DNase-seq / ChIP-seq)
    distributed as `.bigWig`/`.bw`.

    Args:
        agg: "mean" or "sum" (as provided by bigWig summary statistics)
        fill: value used when no data is present for an interval

    Returns:
        y: shape (n_bins,)
    """

    try:
        import pyBigWig  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Binning bigWig tracks requires the optional dependency 'pyBigWig'. "
            "Install with: pip install 'spectral-multiomics-engine[tracks]' (or pip install pyBigWig)."
        ) from e

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    agg_l = str(agg).lower()
    if agg_l not in {"mean", "sum"}:
        raise ValueError(f"Unknown agg={agg!r}; expected 'mean' or 'sum'")

    bw = pyBigWig.open(str(path))
    try:
        chrom = str(bins.chrom)
        if chrom not in bw.chroms():
            raise ValueError(f"Chrom {chrom!r} not found in bigWig: {path}")

        n = int(bins.n_bins)
        y = np.empty(n, dtype=np.float64)

        for b in range(n):
            s = int(bins.start + b * bins.binsize)
            e = int(s + bins.binsize)
            v = bw.stats(chrom, s, e, type=agg_l, exact=False)[0]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                y[b] = float(fill)
            else:
                y[b] = float(v)
    finally:
        try:
            bw.close()
        except Exception:
            pass

    return y.astype(np.float32)
