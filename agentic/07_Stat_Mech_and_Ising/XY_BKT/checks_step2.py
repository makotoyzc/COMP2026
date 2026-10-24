"""Step 2: measure Y(T) at four lattice sizes and extract T_BKT from the jump.

WHAT IS CHECKED
  1. Equilibration: at L = 32, T = 0.92 (right where the runs are slowest) an
     ordered and a random start must give the same Y. If they do not, the runs
     are too short and nothing below means anything.
  2. T*(L), the crossing of Y(T) with 2 T / pi, falls as L grows.
  3. The fit T*(L) = T_BKT + c / (ln L)^2 extrapolates to within 0.02 of the
     accepted 0.8929.

PARALLEL RUNS
-------------
Every (L, T) is an independent chain, so each gets its own process and its own
random stream from SeedSequence.spawn. The big lattices are submitted first so
the slowest jobs are not left to the end.

Run:  ~/anaconda3/bin/python3 checks_step2.py   (about 8 min on 18 cores)
Writes results_step2.npz. Make the figures with plot_step2.py.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from jump import T_BKT, universal, measure, crossing_err, fit_log

LS = [8, 16, 32, 64]
TS = [0.70, 0.75, 0.80, 0.84, 0.86, 0.88, 0.90, 0.92,
      0.94, 0.96, 0.98, 1.00, 1.02, 1.05, 1.10]
N_THERM, N_MEAS = 40000, 200000
EQ = (32, 0.92)  # the equilibration check
WORKERS = 14

if __name__ == "__main__":
    jobs = [(L, T, "ordered") for L in sorted(LS, reverse=True) for T in TS]
    jobs.append((EQ[0], EQ[1], "random"))
    seeds = np.random.SeedSequence(7).spawn(len(jobs))

    print(f"{len(jobs)} runs of {N_THERM} + {N_MEAS} sweeps on {WORKERS} workers")
    res = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(measure, L, T, N_THERM, N_MEAS, seed, start)
                   for (L, T, start), seed in zip(jobs, seeds)]
        for n, f in enumerate(as_completed(futures), 1):
            r = f.result()
            res[(r["L"], r["T"], r["start"])] = r
            print(f"  [{n:2d}/{len(jobs)}] L={r['L']:3d} T={r['T']:.2f} "
                  f"{r['start']:7s} Y = {r['Y']:.4f} +/- {r['dY']:.4f}", flush=True)

    Y = np.array([[res[(L, T, "ordered")]["Y"] for T in TS] for L in LS])
    dY = np.array([[res[(L, T, "ordered")]["dY"] for T in TS] for L in LS])
    nv = np.array([[res[(L, T, "ordered")]["n_vort"] for T in TS] for L in LS])
    dnv = np.array([[res[(L, T, "ordered")]["dn_vort"] for T in TS] for L in LS])

    # ---- Check 1: did the chains forget where they started? ---------------------
    a, b = res[(EQ[0], EQ[1], "ordered")], res[(EQ[0], EQ[1], "random")]
    sig = abs(a["Y"] - b["Y"]) / np.hypot(a["dY"], b["dY"])
    print(f"\ncheck 1  equilibration at L = {EQ[0]}, T = {EQ[1]}")
    print(f"   Y = {a['Y']:.4f} +/- {a['dY']:.4f} (ordered) vs "
          f"{b['Y']:.4f} +/- {b['dY']:.4f} (random)   {sig:.1f} sigma apart   "
          f"{'PASS' if sig < 3 else 'FAIL'}")

    # ---- Check 2: the crossings move down with L --------------------------------
    rng = np.random.default_rng(11)
    Tstar = np.array([crossing_err(TS, Y[i], dY[i], rng) for i in range(len(LS))])
    print("\ncheck 2  crossings Y(T*) = 2 T* / pi")
    for L, (t, dt) in zip(LS, Tstar):
        print(f"   L = {L:3d}:  T* = {t:.4f} +/- {dt:.4f}")
    falls = bool(np.all(np.diff(Tstar[:, 0]) < 0))
    print(f"   T* falls with L: {falls}   {'PASS' if falls else 'FAIL'}")

    # ---- Check 3: extrapolate in 1 / (ln L)^2 -----------------------------------
    T0, dT0, c, dc = fit_log(np.array(LS), Tstar[:, 0], Tstar[:, 1])
    off = abs(T0 - T_BKT)
    print("\ncheck 3  fit T*(L) = T_BKT + c / (ln L)^2")
    print(f"   T_BKT = {T0:.4f} +/- {dT0:.4f}   (accepted {T_BKT})   c = {c:.3f} +/- {dc:.3f}")
    print(f"   off by {off:.4f}   {'PASS' if off < 0.02 else 'FAIL'}")

    np.savez("results_step2.npz", Ls=LS, Ts=TS, Y=Y, dY=dY, n_vort=nv, dn_vort=dnv,
             Tstar=Tstar[:, 0], dTstar=Tstar[:, 1], fit=[T0, dT0, c, dc],
             n_therm=N_THERM, n_meas=N_MEAS,
             **{f"theta_32_{T:.2f}": res[(32, T, "ordered")]["theta"]
                for T in (0.70, 0.90, 1.10)})
    print("\nwrote results_step2.npz")
