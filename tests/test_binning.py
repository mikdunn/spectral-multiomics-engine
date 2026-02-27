import numpy as np

from spectral_multiomics.binning import GenomicBins, bin_bedgraph


def test_bin_bedgraph_roundtrip(tmp_path):
    bins = GenomicBins(chrom="chr1", start=0, binsize=10, n_bins=5)

    # One row per bin; should round-trip exactly under mean aggregation.
    values = np.array([1.0, 2.0, -1.0, 0.0, 5.5], dtype=np.float32)
    lines = []
    for b, v in enumerate(values):
        s = b * bins.binsize
        e = s + bins.binsize
        lines.append(f"chr1\t{s}\t{e}\t{float(v)}\n")

    p = tmp_path / "x.tsv"
    p.write_text("".join(lines))

    y = bin_bedgraph(p, bins, agg="mean")
    assert y.shape == (bins.n_bins,)
    assert np.allclose(y, values, atol=1e-6)


def test_bin_bedgraph_partial_overlap(tmp_path):
    bins = GenomicBins(chrom="chr1", start=0, binsize=10, n_bins=2)

    # interval [0,5) with value 2 should cover half of bin0
    # mean in bin0 should be 2 (since mean is computed over covered bp, not full bin)
    p = tmp_path / "x.tsv"
    p.write_text("chr1\t0\t5\t2\n")

    y = bin_bedgraph(p, bins, agg="mean", fill=0.0)
    assert np.allclose(y, np.array([2.0, 0.0], dtype=np.float32))
