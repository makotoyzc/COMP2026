"""Step 3: the order parameter across melting, with every run in parallel.

Checks on psi6 itself:
  1. A perfect hexagonal crystal has |psi6_j| = 1 for every disk and Psi6 = 1.
  2. Random points (no order at all) give Psi6 of order 1/sqrt(N).

Then the physics. Each eta is run twice, from opposite sides:
  - "crystal": start from a perfect crystal at that eta
  - "liquid":  start from a liquid at eta = 0.60 and squeeze up to that eta
If both starts give the same Psi6, the runs have forgotten where they began and
the answer is trustworthy. If not, the runs are too short for that eta.

PARALLEL RUNS
-------------
Every (eta, start) pair is an independent run, so each goes to its own process
(ProcessPoolExecutor, the Python version of MATLAB's parfeval). Each process
gets its own random stream from SeedSequence.spawn. Handing every process the
same seed would make them identical copies, not independent runs.

Run:  ~/anaconda3/bin/python3 checks_step3.py   (about 10 min on 14 cores)
Writes results_step3.npz. Make the figures with plot_step3.py.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from melting import hex_lattice, psi6, run

ETAS = [0.66, 0.68, 0.69, 0.70, 0.71, 0.72, 0.74]
SWEEPS = {0.66: 20000, 0.74: 20000}  # every other eta gets 60000: it is near the transition
STARTS = ["crystal", "liquid"]

if __name__ == "__main__":
    nx, ny = 16, 18
    N = nx * ny

    # ---- Check 1: perfect crystal -----------------------------------------------
    x, L = hex_lattice(nx, ny, 0.70)
    assert np.isclose(N * np.pi / 4 / (L[0] * L[1]), 0.70)
    p = psi6(x, L)
    assert np.allclose(np.abs(p), 1) and np.isclose(abs(p.mean()), 1)
    print("check 1  perfect crystal: |psi6| = 1 for all disks, Psi6 = 1   PASS")

    # ---- Check 2: random points ---------------------------------------------------
    rng = np.random.default_rng(3)
    Psi = [abs(psi6(rng.uniform(0, 1, (N, 2)) * L, L).mean()) for _ in range(200)]
    ratio = np.sqrt(np.mean(np.square(Psi)) * N)
    assert 0.3 < ratio < 3, ratio
    print(f"check 2  random points: Psi6 * sqrt(N) = {ratio:.2f}, order 1      PASS")

    # ---- All runs at once ---------------------------------------------------------
    jobs = [(eta, start) for eta in ETAS for start in STARTS]
    seeds = np.random.SeedSequence(3).spawn(len(jobs))
    results = {}
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run, eta, start, SWEEPS.get(eta, 60000), seed)
                   for (eta, start), seed in zip(jobs, seeds)]
        for f in as_completed(futures):
            r = f.result()
            results[(r["eta"], r["start"])] = r
            print(f"  done: eta = {r['eta']:.2f} from {r['start']}", flush=True)

    # ---- Do the two starts agree? ---------------------------------------------------
    print()
    print("  eta    from crystal     from liquid      (mean +/- spread over the 2nd half)")
    for eta in ETAS:
        tails = [results[(eta, s)]["series"][len(results[(eta, s)]["series"]) // 2:] for s in STARTS]
        (m1, m2), (s1, s2) = [t.mean() for t in tails], [t.std() for t in tails]
        verdict = "agree" if abs(m1 - m2) < s1 + s2 else "DISAGREE"
        print(f"  {eta:.2f}   {m1:.2f} +/- {s1:.2f}     {m2:.2f} +/- {s2:.2f}     {verdict}")

    save = {}
    for (eta, start), r in results.items():
        tag = f"{start}_{eta:.2f}"
        save[tag + "_series"] = r["series"]
        save[tag + "_x"] = r["x"]
        save[tag + "_L"] = r["L"]
    np.savez("results_step3.npz", etas=ETAS, **save)
    print("wrote results_step3.npz")
