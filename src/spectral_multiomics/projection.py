from __future__ import annotations

import numpy as np


def project_signal(evecs: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Project binned signal(s) onto the spectral basis.

    Args:
        evecs: (n_bins, k)
        signal: (n_bins,) or (n_bins, m)

    Returns:
        coeffs: (k,) or (k, m)
    """
    E = np.asarray(evecs, dtype=np.float32)
    y = np.asarray(signal, dtype=np.float32)

    if E.ndim != 2:
        raise ValueError("evecs must be 2D (n_bins, k)")

    if y.ndim == 1:
        if y.shape[0] != E.shape[0]:
            raise ValueError("signal length must match evecs rows")
        return (E.T @ y).astype(np.float32)

    if y.ndim == 2:
        if y.shape[0] != E.shape[0]:
            raise ValueError("signal first dimension must match evecs rows")
        return (E.T @ y).astype(np.float32)

    raise ValueError("signal must be 1D or 2D")
