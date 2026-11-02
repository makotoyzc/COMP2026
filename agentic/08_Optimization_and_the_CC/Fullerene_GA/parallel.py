"""Run independent jobs on all CPU cores.

Import this module BEFORE numpy: it pins each process to one BLAS thread, because on
240x240 matrices extra threads buy nothing (2.5 ms vs 2.1 ms per call at N = 60) and 18
processes x 18 threads would fight over the cores. All the speed-up comes from running
18 independent processes.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import time
from concurrent.futures import ProcessPoolExecutor

WORKERS = os.cpu_count()


def run(fn, tasks, label="", chunksize=1):
    """[fn(t) for t in tasks], spread over WORKERS processes, in order."""
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        out = list(ex.map(fn, tasks, chunksize=chunksize))
    print(f"{label}: {len(tasks)} jobs on {WORKERS} processes in {time.perf_counter() - t0:.1f} s")
    return out
