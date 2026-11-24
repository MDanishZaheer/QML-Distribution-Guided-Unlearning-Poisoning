"""Run the unchanged historical analysis with cached, already verified metadata."""

from time import monotonic
from unittest.mock import patch

from src.evaluation import analysis
from src.artifacts import load_json


def cached_phase1_tables(conditions, review=None, progress=None):
    """Avoid hundreds of repeated small storage reads; never fit or predict models."""
    cache = {}
    start = monotonic()
    def read(path):
        if path not in cache:
            local = review._read_path(path) if review is not None else path
            cache[path] = load_json(local)
        return cache[path]
    if progress is not None:
        progress(f"Building tables for {len(conditions)} conditions using cached metadata.")
    with patch.object(analysis, "load_json", read):
        tables = analysis.phase1_tables(conditions)
    if progress is not None:
        progress(f"Tables ready in {monotonic() - start:.1f}s; {len(cache)} unique metadata files read.")
    return tables
