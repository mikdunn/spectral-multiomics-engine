    pe = sub.add_parser(
        "demo-encode",
        help=(
            "Download ENCODE GM12878 ATAC-seq bigWig (GRCh38), bin to match a GRCh38 .cool contact map, and run the pipeline."
        ),
    )
    pe.add_argument("--data_dir", type=str, default="data/real/encode_demo")
    pe.add_argument("--out_dir", type=str, default="outputs/demo_encode")
    pe.add_argument("--overwrite", action="store_true", help="Re-download/regenerate demo inputs")
    pe.add_argument("--k", type=int, default=8)
    pe.add_argument("--normalized_laplacian", action=argparse.BooleanOptionalAction, default=True)
    pe.add_argument("--include_trivial", action=argparse.BooleanOptionalAction, default=False)
    pe.add_argument("--agg", type=str, default="mean", choices=["mean", "sum"])
    if args.cmd == "demo-encode":
        from .realdata import download_encode_bigwig_gm12878_grch38, DEFAULT_COOLER_YEAST_URL
        import shutil
        from .binning import GenomicBins, bin_bigwig
        import numpy as np
        import cooler

        data_dir = Path(args.data_dir)
        ensure_dir(data_dir)

        # Download small GRCh38 .cool contact map (Cooler test file)
        cool_url = "https://github.com/open2c/cooler/raw/refs/heads/master/tests/data/dec2_20_pluslig_1pGene_grch38_UBR4_D_1nt.pairwise.sorted.cool"
        cool_path = data_dir / "contact_grch38.cool"
        if not cool_path.exists() or args.overwrite:
            tmp = cool_path.with_suffix(".tmp")
            with urllib.request.urlopen(cool_url) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(cool_path)

        # Download ENCODE GM12878 ATAC-seq bigWig (GRCh38)
        bw_path = data_dir / "GM12878_ATAC_GRCh38.bigWig"
        download_encode_bigwig_gm12878_grch38(bw_path, overwrite=args.overwrite)

        # Get bins from .cool file
        c = cooler.Cooler(str(cool_path))
        chrom = str(c.chromnames[0])
        bins_df = c.bins().fetch(chrom)
        n_bins = bins_df.shape[0]
        binsize = int(bins_df.iloc[0]["end"] - bins_df.iloc[0]["start"])
        bins = GenomicBins(chrom=chrom, start=int(bins_df.iloc[0]["start"]), binsize=binsize, n_bins=n_bins)

        # Bin bigWig signal to match contact map bins
        y = bin_bigwig(bw_path, bins, agg=args.agg)
        signal_path = data_dir / f"signal.{chrom}.binned.npy"
        np.save(signal_path, y)

        # Run pipeline
        out = run_pipeline(
            contact=cool_path,
            signal=signal_path,
            out_dir=args.out_dir,
            binsize=binsize,
            chrom=chrom,
            k=int(args.k),
            normalized_laplacian=bool(args.normalized_laplacian),
            include_trivial=bool(args.include_trivial),
            agg=str(args.agg),
        )
        print("Demo data:")
        print("  contact:", cool_path.as_posix())
        print("  signal :", signal_path.as_posix())
        print("Wrote outputs to:", out.out_dir.as_posix())
        return
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline
from .realdata import DEFAULT_COOLER_YEAST_URL, prepare_cooler_demo_dataset
from .synth import synth_dataset


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spectral-multiomics")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("synth", help="Generate a small synthetic contact map + bedGraph signal")
    ps.add_argument("--out_dir", type=str, default="data/synthetic")
    ps.add_argument("--n_bins", type=int, default=256)
    ps.add_argument("--binsize", type=int, default=100_000)
    ps.add_argument("--chrom", type=str, default="chr1")
    ps.add_argument("--seed", type=int, default=0)

    pr = sub.add_parser("run", help="Run spectral projection pipeline")
    pr.add_argument("--contact", type=str, required=True)
    pr.add_argument("--signal", type=str, required=True)
    pr.add_argument("--out_dir", type=str, required=True)
    pr.add_argument("--binsize", type=int, required=True)
    pr.add_argument("--chrom", type=str, default="chr1")
    pr.add_argument("--k", type=int, default=16)
    pr.add_argument("--normalized_laplacian", action=argparse.BooleanOptionalAction, default=True)
    pr.add_argument("--include_trivial", action=argparse.BooleanOptionalAction, default=False)
    pr.add_argument("--agg", type=str, default="mean", choices=["mean", "sum"])

    pd = sub.add_parser(
        "demo-cooler",
        help=(
            "Download a small real Hi-C Cooler (.cool) file, generate a matching bedGraph signal, "
            "and run the pipeline (end-to-end demo)."
        ),
    )
    pd.add_argument("--data_dir", type=str, default="data/real/cooler_yeast_demo")
    pd.add_argument("--out_dir", type=str, default="outputs/demo_cooler")
    pd.add_argument("--url", type=str, default=DEFAULT_COOLER_YEAST_URL)
    pd.add_argument("--chrom", type=str, default=None)
    pd.add_argument("--k", type=int, default=16)
    pd.add_argument("--normalized_laplacian", action=argparse.BooleanOptionalAction, default=True)
    pd.add_argument("--include_trivial", action=argparse.BooleanOptionalAction, default=False)
    pd.add_argument("--agg", type=str, default="mean", choices=["mean", "sum"])
    pd.add_argument("--overwrite", action="store_true", help="Re-download/regenerate demo inputs")

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.cmd == "synth":
        paths = synth_dataset(
            out_dir=args.out_dir,
            n_bins=int(args.n_bins),
            binsize=int(args.binsize),
            chrom=str(args.chrom),
            seed=int(args.seed),
        )
        print("Wrote:")
        for k, v in paths.items():
            print(f"  {k}: {Path(v).as_posix()}")
        return

    if args.cmd == "run":
        out = run_pipeline(
            contact=args.contact,
            signal=args.signal,
            out_dir=args.out_dir,
            binsize=int(args.binsize),
            chrom=str(args.chrom),
            k=int(args.k),
            normalized_laplacian=bool(args.normalized_laplacian),
            include_trivial=bool(args.include_trivial),
            agg=str(args.agg),
        )
        print("Wrote outputs to:", out.out_dir.as_posix())
        return

    if args.cmd == "demo-cooler":
        demo = prepare_cooler_demo_dataset(
            out_dir=args.data_dir,
            url=str(args.url),
            chrom=args.chrom,
            overwrite=bool(args.overwrite),
        )

        out = run_pipeline(
            contact=demo.cool_path,
            signal=demo.signal_path,
            out_dir=args.out_dir,
            binsize=int(demo.binsize),
            chrom=str(demo.chrom),
            k=int(args.k),
            normalized_laplacian=bool(args.normalized_laplacian),
            include_trivial=bool(args.include_trivial),
            agg=str(args.agg),
        )

        print("Demo data:")
        print("  contact:", demo.cool_path.as_posix())
        print("  signal :", demo.signal_path.as_posix())
        print("  meta   :", demo.meta_path.as_posix())
        print("Wrote outputs to:", out.out_dir.as_posix())
        return

    raise SystemExit(f"Unknown command: {args.cmd}")
