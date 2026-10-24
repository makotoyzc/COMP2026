"""Step 1 checks: is the XY sampler correct?

  1. Low T (spin waves): E/N = -2 + T/2 and Y = 1 - T/4, up to O(T^2).
  2. High T: E/N = -2 (u + 2 u^3), u = I1(1/T) / I0(1/T), up to O(u^5).
  3. Net vorticity is exactly zero in every configuration (a torus has no
     boundary for a vortex to escape through). No vortices at T = 0.1, many
     at T = 1.5.
  4. Ordered and random starts agree at T = 1.2.

Run:  ~/anaconda3/bin/python3 checks_step1.py   (about 1 min)
Writes figures/step1.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0, i1

from xy import run, blocks, stiffness, mean_err

rng = np.random.default_rng(1)
L = 16
N = L * L
n_therm, n_meas = 2000, 20000


def measure(T, start="ordered"):
    rec = run(L, T, n_therm, n_meas, rng, start)
    bl = blocks(rec)
    E = mean_err([b["E"].mean() / N for b in bl])
    Y = mean_err([stiffness(b, T, N) for b in bl])
    return rec, E, Y


def high_T(T):
    u = i1(1 / T) / i0(1 / T)
    return -2 * (u + 2 * u ** 3), -2 * u


q_all_zero = True

# ---- Check 1: low T ----------------------------------------------------------
print("check 1  low T, spin waves")
for T in (0.1, 0.2):
    rec, (E, dE), (Y, dY) = measure(T)
    q_all_zero &= bool(np.all(rec["q"] == 0))
    tol = 0.3 * T ** 2
    okE = abs(E - (-2 + T / 2)) < 3 * dE + tol
    okY = abs(Y - (1 - T / 4)) < 3 * dY + tol
    print(f"   T={T}:  E/N = {E:.5f} +/- {dE:.5f}  vs {-2 + T / 2:.5f}   "
          f"Y = {Y:.5f} +/- {dY:.5f}  vs {1 - T / 4:.5f}   (tol {tol:.4f})  "
          f"{'PASS' if okE and okY else 'FAIL'}")

# ---- Check 2: high T ---------------------------------------------------------
print("check 2  high T, series expansion")
for T in (5.0, 10.0):
    rec, (E, dE), _ = measure(T, start="random")
    q_all_zero &= bool(np.all(rec["q"] == 0))
    with_plaq, bond_only = high_T(T)
    ok = abs(E - with_plaq) < 4 * dE + 2e-4
    print(f"   T={T}:  E/N = {E:.5f} +/- {dE:.5f}  vs {with_plaq:.5f}  "
          f"(bond only {bond_only:.5f})  {'PASS' if ok else 'FAIL'}")

# ---- Check 3: vortices -------------------------------------------------------
print("check 3  vortices")
rec_cold, _, _ = measure(0.1)
rec_hot, _, _ = measure(1.5, start="random")
q_all_zero &= bool(np.all(rec_cold["q"] == 0) and np.all(rec_hot["q"] == 0))
cold = rec_cold["n_vort"].mean() / N
hot = rec_hot["n_vort"].mean() / N
print(f"   net vorticity zero in every sweep of every run: {q_all_zero}")
print(f"   vortex density  T=0.1: {cold:.5f}   T=1.5: {hot:.4f}  "
      f"{'PASS' if q_all_zero and cold == 0 and hot > 0.02 else 'FAIL'}")

# ---- Check 4: two starts -----------------------------------------------------
print("check 4  ordered vs random start at T = 1.2")
_, (Eo, dEo), (Yo, dYo) = measure(1.2, "ordered")
_, (Er, dEr), (Yr, dYr) = measure(1.2, "random")
okE = abs(Eo - Er) < 3 * np.hypot(dEo, dEr)
okY = abs(Yo - Yr) < 3 * np.hypot(dYo, dYr)
print(f"   E/N: {Eo:.4f} +/- {dEo:.4f} vs {Er:.4f} +/- {dEr:.4f}   "
      f"Y: {Yo:.4f} +/- {dYo:.4f} vs {Yr:.4f} +/- {dYr:.4f}   "
      f"{'PASS' if okE and okY else 'FAIL'}")

# ---- Figure: E/N(T) between the two limits -----------------------------------
Ts = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.6, 2.0, 3.0, 5.0])
E_T = []
for T in Ts:
    _, (E, dE), _ = measure(T, start="ordered" if T < 1 else "random")
    E_T.append((E, dE))
E_T = np.array(E_T)

t_lo = np.linspace(0.05, 1.2, 100)
t_hi = np.linspace(1.0, 5.0, 100)
plt.errorbar(Ts, E_T[:, 0], yerr=E_T[:, 1], fmt="o", label=f"Monte Carlo, L = {L}")
plt.plot(t_lo, -2 + t_lo / 2, "--", label="spin waves: -2 + T/2")
plt.plot(t_hi, high_T(t_hi)[0], ":", label="high-T series")
plt.axvline(0.8929, color="gray", lw=0.8)
plt.xlabel("T / J")
plt.ylabel("E / N")
plt.legend()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/step1.png", dpi=150)
print("wrote figures/step1.png")
