"""Memory-aware execution limits for the GPU actually allocated by runtime."""

import os
from dataclasses import asdict, replace


def benchmark_policy(candidates, *, automatic=False):
    """Pin selected execution across sessions despite fluctuating free VRAM.

Hardware/driver/package identity is still checked by each benchmark. The first
passing auto selection retains its exact recorded memory cap and worker count.
"""
    return [{key: value for key, value in asdict(candidate).items()
             if not (automatic and key == "memory_mb")} for candidate in candidates]


def memory_budget_mb(sample, fraction=0.8):
    if not 0 < fraction < 1:
        raise ValueError("Keep a positive GPU memory reserve.")
    total = float(sample.get("memory.total", 20480))
    free = float(sample.get("memory.free", total))
    return max(512, int(min(total * fraction, max(512, free - 1024))))


def parallel_worker_counts():
    """Include all five folds and higher concurrency; benchmark rather than assume."""
    return (1, 2, 4, 5, 6, 8, 12, 16, 24)


def aer_cpu_threads():
    """Permit more allocated CPU cores while keeping small-circuit overhead bounded."""
    return min(8, os.cpu_count() or 1)


def execution_refinements(best):
    """Refine only the fastest passing candidate, avoiding another full Cartesian grid."""
    cores = os.cpu_count() or 1
    return [replace(best, threads=n) for n in (1, 4) if n <= cores and n != best.threads] + [
        replace(best, gather_ms=10.0)]


def host_resources():
    try:
        available = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        available = None
    return {"cpu_count": os.cpu_count(), "available_host_memory_bytes": available}
