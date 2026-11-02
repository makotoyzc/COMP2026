"""Step 6: the genetic algorithm (ga.py) on C20 and C60, from random atoms.

Usage:  python3 step6_ga.py N mu seed1 [seed2 ...] [--matings M]
        e.g.  python3 step6_ga.py 60 0 1           one C60 ecology, no mutation
              python3 step6_ga.py 20 0 1 2 3 4     four C20 ecologies
Each ecology uses all cores and writes ga_runs/C{N}_mu{mu}_seed{seed}.npz.
For C20 the target is the bowl; for C60 it is E_ref. A run stops when it hits the target.
"""
import parallel  # first: one BLAS thread per process

import argparse
import os

import numpy as np

import carbon_tb as tb
import structures as st
import ga


def targets():
    s4 = np.load("step4_results.npz", allow_pickle=True)
    return {20: float(s4["E20"][1]), 60: float(s4["E_ref"])}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("N", type=int)
    ap.add_argument("mu", type=float)
    ap.add_argument("seeds", type=int, nargs="+")
    ap.add_argument("--matings", type=int, default=6000)
    args = ap.parse_args()
    E_target = targets()[args.N]
    os.makedirs("ga_runs", exist_ok=True)
    for seed in args.seeds:
        print(f"===== C{args.N}, mu = {args.mu}, seed {seed}: target {E_target:+.4f} eV/atom =====",
              flush=True)
        h = ga.run(args.N, seed, args.mu, matings=args.matings, E_target=E_target)
        R = h["pop_R"][int(np.argmin(h["pop_E"]))]
        ok = min(h["pop_E"]) < E_target + 1e-3
        at = h["best_at"][-1] if h["best_at"] else 0
        print(f"  {'FOUND' if ok else 'not found'}: best {min(h['pop_E']):+.4f} eV/atom"
              f" ({min(h['pop_E']) - E_target:+.4f} from target) after {h['matings']} matings,"
              f" {h['calls'][-1]} calls total; last improvement at mating {at};"
              f" {h['accepted']} children accepted; {h['seconds']:.0f} s")
        print(f"  best: {st.classify(R)}, {st.describe(R)}", flush=True)
        np.savez(f"ga_runs/C{args.N}_mu{args.mu}_seed{seed}.npz",
                 N=args.N, mu=args.mu, seed=seed, found=ok, E_target=E_target,
                 calls=np.array(h["calls"]), Emin=np.array(h["Emin"]), Emax=np.array(h["Emax"]),
                 best=np.array(h["best"]), best_E=np.array(h["best_E"]),
                 best_at=np.array(h["best_at"]), init_calls=h["init_calls"],
                 matings=h["matings"], seconds=h["seconds"])
