"""Gate 1 (PLAN.md): checks of the energy in carbon_tb.py.

Usage:  python3 gate1.py      -> prints the checks, saves gate1_bulk.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

import carbon_tb as tb


def diamond(b):
    """fcc primitive cell, 2 atoms, bond length b."""
    a = 4 * b / np.sqrt(3)
    lat = np.array([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
    return lat, np.array([[0, 0, 0], [a / 4, a / 4, a / 4]])


def graphene(b):
    """Single sheet (= graphite in this model), 2 atoms, bond length b, 20 A of vacuum."""
    a = np.sqrt(3) * b
    lat = np.array([[a, 0, 0], [a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 20.0]])
    return lat, np.array([[0, 0, 0], (lat[0] + lat[1]) / 3])


def e_bulk(struct, b, n):
    lat, basis = STRUCT[struct](b)
    return tb.binding(tb.energy_periodic(lat, basis, KMESH[struct](n)), len(basis))


STRUCT = dict(diamond=diamond, graphene=graphene)
KMESH = dict(diamond=lambda n: (n, n, n), graphene=lambda n: (n, n, 1))


def random_cluster(N, rng, box, dmin=1.2):
    """N points in a cube of side box, no two closer than dmin."""
    R = []
    while len(R) < N:
        p = rng.uniform(0, box, 3)
        if all(np.linalg.norm(p - q) >= dmin for q in R):
            R.append(p)
    return np.array(R)


rng = np.random.default_rng(8)

print("===== 0. cutoff tails vs Table 2 of Xu et al. (1992) =====")
TABLE2 = dict(s=[6.7392620074314e-3, -8.1885359517898e-2, 0.1932365259144, 0.3542874332380],
              phi=[2.2504290109e-8, -1.4408640561e-6, 2.1043303374e-5, 6.6024390226e-5])
for nm, g, r1 in (("s", tb.s_r, tb.R1), ("phi", tb.phi_r, tb.D1)):
    rel = np.abs(g.tail / TABLE2[nm] - 1).max()
    print(f"{nm:3s} tail from r1 = {r1} A, built here: " + " ".join(f"{c:+.10e}" for c in g.tail))
    print(f"    Table 2:                      " + " ".join(f"{c:+.10e}" for c in TABLE2[nm])
          + f"   max rel. diff {rel:.1e}")
    lo, hi = g(r1 - 1e-9), g(r1 + 1e-9)
    print(f"    jump at r1: value {hi[0] - lo[0]:+.1e}, slope {hi[1] - lo[1]:+.1e};"
          f"  at RM: {g(tb.RM - 1e-9)[0]:+.1e}, slope {g(tb.RM - 1e-9)[1]:+.1e}")

print("\n===== 1. free atoms =====")
R = 10.0 * np.array(list(np.ndindex(3, 3, 2)), float)      # 18 atoms on a 10 A grid
Eb = tb.binding(tb.energy(R), len(R))
print(f"18 atoms 10 A apart: binding = {Eb:+.2e} eV/atom   (expect 0)")
print(f"free-atom energy E_ATOM = 2Es + 2Ep + C0 = {tb.E_ATOM:+.6f} eV")

print("\n===== k-mesh convergence at b = 1.54 / 1.42 A =====")
for struct, b, ns in (("diamond", 1.54, (4, 6, 8, 10, 12, 16)),
                      ("graphene", 1.42, (6, 12, 18, 24, 36, 48, 60))):
    prev = None
    for n in ns:
        e = e_bulk(struct, b, n)
        dif = "" if prev is None else f"   change {e - prev:+.1e}"
        print(f"{struct:8s} mesh {n:2d}: {e:+.6f} eV/atom{dif}")
        prev = e
NK = dict(diamond=12, graphene=48)

print("\n===== 2-4. bulk minima =====")
res = {}
for struct, b0 in (("diamond", 1.54), ("graphene", 1.42)):
    opt = minimize_scalar(lambda b: e_bulk(struct, b, NK[struct]),
                          bounds=(b0 - 0.1, b0 + 0.1), method="bounded",
                          options=dict(xatol=1e-5))
    res[struct] = (opt.x, opt.fun)
    print(f"{struct:8s} (mesh {NK[struct]}): b_min = {opt.x:.4f} A, "
          f"binding = {opt.fun:+.4f} eV/atom")
print(f"check 2: diamond b_min - 1.54 = {res['diamond'][0] - 1.54:+.4f} A   (|.| <= 0.01)")
print(f"check 3: graphite b_min - 1.42 = {res['graphene'][0] - 1.42:+.4f} A   (|.| <= 0.01)")
for struct in res:
    lat, basis = STRUCT[struct](res[struct][0])
    i, j, d, r = tb.periodic_terms(lat, np.asarray(basis))
    print(f"{struct:8s}: x = sum_j phi = {np.bincount(i, tb.phi_r(r)[0]).max():.1f} eV"
          f"   (polynomial f used as published while x <= X_LIN = {tb.X_LIN:.0f})")
for struct in res:
    print(f"{struct:8s}: E_tot/N = {res[struct][1] + tb.E_ATOM:+.4f} eV/atom   (paper Fig. 1 plots this: ~ -8.4)")
dE = res["graphene"][1] - res["diamond"][1]
print(f"check 4: E(graphite) - E(diamond) = {dE:+.4f} eV/atom   (|.| <~ 0.1)")

print("\n===== 5. rotation + translation invariance =====")
R = random_cluster(20, rng, box=5.0)
Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
R2 = R @ Q.T + rng.uniform(-10, 10, 3)
E1, E2 = tb.energy(R), tb.energy(R2)
print(f"20-atom random cluster: E = {E1:+.10f} eV, rotated+shifted differs by {E2 - E1:+.1e} eV"
      f"   (|.| <= 1e-10)")
print(f"(bonds shorter than RM: {len(tb.pairs(R)[0])})")

# Figure: binding energy vs bond length
fig, ax = plt.subplots(figsize=(5.5, 4))
for struct, label in (("diamond", "diamond"), ("graphene", "graphite (one sheet)")):
    bs = np.linspace(1.30, 1.75, 31)
    ax.plot(bs, [e_bulk(struct, b, NK[struct]) for b in bs], label=label)
    ax.plot(*res[struct], "ko", ms=4)
ax.set_xlabel("Bond length (Å)")
ax.set_ylabel("Binding energy (eV/atom)")
ax.legend()
fig.savefig("gate1_bulk.png", dpi=200, bbox_inches="tight")
print("\nsaved gate1_bulk.png")
