"""Utilities for reproducing AV-SyncBench evaluations."""

from .manifest import ManifestRecord, iter_manifest
from .metrics import PairwiseResult, summarize_results

__all__ = [
    "ManifestRecord",
    "PairwiseResult",
    "iter_manifest",
    "summarize_results",
]

