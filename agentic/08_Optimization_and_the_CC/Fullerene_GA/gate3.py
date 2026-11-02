"""Gate 3 (PLAN.md): relax the ideal truncated icosahedron; this is E_ref for every search.

Usage:  python3 gate3.py      -> prints the checks, saves c60_ref.npy
"""
import numpy as np

import carbon_tb as tb
import structures as st

R0 = st.c60_ideal(1.42)
R, E, info = st.relax(R0, fmax=1e-6)
print(f"relaxed from the 1.42 A ideal cage: {info['nit']} iterations, {info['nfev']} calls,"
      f" max |F| = {info['fmax']:.1e} eV/A, converged = {info['converged']}")
np.save("c60_ref.npy", R)

print("\n===== 1. coordination =====")
z = st.coordination(R, cut=1.6)
i, j, r = st.bonds(R, cut=1.6)
print(f"neighbors within 1.6 A: min {z.min()}, max {z.max()}   (expect 3, 3); bonds = {len(r)} (90)")

print("\n===== 3. rings =====")
nr, found = st.rings(R)
print(f"rings: {nr}   (expect 12 pentagons, 20 hexagons, nothing else)")

print("\n===== 2. bond lengths =====")
in_pent = np.array([any({a, b} <= ring for ring in found[5]) for a, b in zip(i, j)])
for name, m in (("pentagon edges (5-6, single)", in_pent), ("hexagon-hexagon (6-6, double)", ~in_pent)):
    print(f"{name:30s}: {m.sum():2d} bonds, length {r[m].mean():.4f} A, spread {np.ptp(r[m]):.1e}")
print(f"single - double = {r[in_pent].mean() - r[~in_pent].mean():+.4f} A   (expect > 0)")

print("\n===== 4. energy =====")
Eb = tb.binding(E, 60)
print(f"binding energy = {Eb:+.4f} eV/atom")
print(f"above graphite (-7.2563 eV/atom, Gate 1): {Eb + 7.2563:+.4f} eV/atom   (a few tenths)")
print(f"x = sum_j phi = {tb.rep_x(R).max():.1f} eV   (<= X_LIN = {tb.X_LIN:.0f}: f as published)")
print(f"radius {np.linalg.norm(R - R.mean(0), axis=1).mean():.4f} A, "
      f"spread {np.ptp(np.linalg.norm(R - R.mean(0), axis=1)):.1e} (all atoms equivalent)")

print("\n===== 5. HOMO-LUMO gap =====")
eps, gap = tb.levels(R)
homo, lumo = eps[119], eps[120]
deg = lambda e: int(np.sum(np.abs(eps - e) < 1e-6))
print(f"HOMO {homo:+.4f} eV (x{deg(homo)}), LUMO {lumo:+.4f} eV (x{deg(lumo)}), gap = {gap:.4f} eV"
      f"   (C60 HOMO is h_u, 5-fold; LUMO t_1u, 3-fold)")
