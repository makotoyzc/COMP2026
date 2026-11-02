"""Gate 2 (PLAN.md): analytic forces vs finite differences, zero net force, timing.

Usage:  python3 gate2.py
"""
import time

import numpy as np

import carbon_tb as tb
import structures as st

rng = np.random.default_rng(2)

print("===== 1. forces vs central finite differences (h = 1e-5 A) =====")
h = 1e-5
for N, radius, dmin in ((10, 2.2, 1.2), (10, 2.6, 1.2), (24, 3.2, 1.2), (12, 1.6, 0.8)):
    R = st.random_cluster(N, rng, radius, dmin)
    E, F = tb.energy_forces(R)
    Fnum = np.zeros_like(R)
    for a in range(N):
        for k in range(3):
            Rp, Rm = R.copy(), R.copy()
            Rp[a, k] += h
            Rm[a, k] -= h
            Fnum[a, k] = -(tb.energy(Rp) - tb.energy(Rm)) / (2 * h)
    _, _, _, r = tb.pairs(R)
    _, gap = tb.levels(R)
    x = tb.rep_x(R)
    print(f"N = {N:2d}, dmin {dmin}: {len(r)} pairs < RM ({np.sum(r >= tb.R1)} in the tail),"
          f" {np.sum(x > tb.X_LIN)} atoms with x > X_LIN, gap = {gap:.3f} eV,")
    print(f"        "
          f" max |F| = {np.abs(F).max():6.2f} eV/A,"
          f" max |F - F_fd| = {np.abs(F - Fnum).max():.1e} eV/A   (<= 1e-6)")
    print(f"        energy_forces E - energy E = {E - tb.energy(R):+.1e} eV,"
          f"   |sum F| = {np.abs(F.sum(0)).max():.1e}   (<= 1e-10)")

R = st.random_cluster(24, rng, 3.2)
E, F = tb.energy_forces(R)
print("\n===== 2. torque (rotation invariance) =====")
tau = np.cross(R, F).sum(0)
print(f"N = 24 cluster: |sum R x F| = {np.abs(tau).max():.1e} eV   (<= 1e-10)")

print("\n===== 3. timing, one energy + force call at N = 60 =====")
for name, R in (("ideal C60", st.c60_ideal()), ("random 60", st.random_cluster(60, rng, 4.6))):
    n = 50
    t0 = time.perf_counter()
    for _ in range(n):
        tb.energy_forces(R)
    dt = (time.perf_counter() - t0) / n
    print(f"{name:10s}: {len(tb.pairs(R)[0]):4d} pairs < RM, {1e3 * dt:.1f} ms per call")
