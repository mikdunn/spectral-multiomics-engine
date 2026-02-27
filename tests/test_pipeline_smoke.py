import numpy as np

from spectral_multiomics.pipeline import run_pipeline
from spectral_multiomics.synth import synth_dataset


def test_pipeline_smoke(tmp_path):
    synth_dir = tmp_path / "synth"
    paths = synth_dataset(synth_dir, n_bins=64, binsize=100, seed=0)

    out_dir = tmp_path / "out"
    out = run_pipeline(
        contact=paths["contact"],
        signal=paths["signal"],
        out_dir=out_dir,
        binsize=100,
        k=8,
    )

    evecs = np.load(out.evecs_path)
    proj = np.load(out.projection_path)
    y = np.load(out.signal_binned_path)

    assert evecs.shape == (64, 8)
    assert y.shape == (64,)
    assert proj.shape == (8,)
