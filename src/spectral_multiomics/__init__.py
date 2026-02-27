"""Spectral multi-omics engine (MVP).

Core idea: contact-map Laplacian eigenvectors provide a structural basis; genomic signals are binned and projected.
"""

from .binning import GenomicBins, bin_bedgraph
from .contact_map import load_contact_map
from .laplacian import laplacian_eigenvectors
from .projection import project_signal

__all__ = [
    "GenomicBins",
    "bin_bedgraph",
    "load_contact_map",
    "laplacian_eigenvectors",
    "project_signal",
]
