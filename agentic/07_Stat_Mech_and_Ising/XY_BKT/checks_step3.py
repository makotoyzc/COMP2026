"""Step 3: solve the amplitude half, then hand it to the phase half.

Mean field theory gives the superconducting gap and a transition temperature
Tc_MF. In 2D that temperature is not where the resistance goes away, because
the phase still has to stop unbinding its vortices. This step computes both and
compares them, with one knob: the normal-state sheet resistance.

WHAT IS CHECKED
  1. Weak coupling reproduces BCS: 2 Delta(0) / Tc_MF -> 3.528 as U falls.
  2. Near Tc the gap follows the BCS square root: Delta / (Tc sqrt(1 - T/Tc))
     extrapolates to 3.06 as T -> Tc (the law is asymptotic).
  3. Above Tc the gap equation has no superconducting root at all.
  4. Limits of the bridge: as R_sq / R_Q -> 0 the stiffness diverges and
     T_BKT -> Tc_MF (mean field becomes right); the lattice coefficient 0.8929
     measured in step 2 always puts T_BKT below the bare Nelson-Kosterlitz
     estimate, and T_BKT / Tc falls monotonically with sheet resistance.

Run:  ~/anaconda3/bin/python3 checks_step3.py   (about 1 min)
Writes results_step3.npz. Make the figures with plot_step3.py.
"""
import numpy as np

from bcs import NK, LATTICE, xi, solve_gap, Tc_mf, gap_curve, stiffness, T_bkt

T, MU, NKGRID = 1.0, -1.0, 256
U = 1.6                       # chosen so Tc_MF is a workable fraction of t
R_RATIOS = np.geomspace(0.05, 20.0, 40)

xi_k = xi(T, MU, NKGRID)

# ---- Check 1: weak coupling is BCS -------------------------------------------
print("check 1  weak coupling: 2 Delta(0) / Tc -> 3.528")
ratios = []
for u in (1.6, 1.4, 1.2, 1.0):
    tc, d0 = Tc_mf(u, xi_k), solve_gap(1e-6, u, xi_k)
    ratios.append(2 * d0 / tc)
    print(f"   U = {u}:  Tc_MF = {tc:.5f}  Delta(0) = {d0:.5f}  ratio = {ratios[-1]:.3f}")
ok1 = max(abs(np.array(ratios) - 3.528)) < 0.02
print(f"   all within 0.02 of 3.528   {'PASS' if ok1 else 'FAIL'}")

# ---- Check 2: the BCS square root near Tc ------------------------------------
Tc = Tc_mf(U, xi_k)
Ts, Deltas = gap_curve(U, xi_k, Tc)
eps = 1 - Ts / Tc
near = (eps > 1e-3) & (eps < 0.10)
shape = Deltas[near] / (Tc * np.sqrt(eps[near]))
slope, limit = np.polyfit(eps[near], shape, 1)
ok2 = abs(limit - 3.06) < 0.02
print(f"\ncheck 2  Delta / (Tc sqrt(1 - T/Tc)) -> {limit:.3f} as T -> Tc "
      f"(it is {shape.min():.3f} at 1 - T/Tc = 0.1; the BCS square root is only "
      f"asymptotic)\n   vs BCS 3.06   {'PASS' if ok2 else 'FAIL'}")

# ---- Check 3: nothing above Tc -----------------------------------------------
above = [solve_gap(f * Tc, U, xi_k) for f in (1.001, 1.1, 2.0)]
ok3 = all(d == 0.0 for d in above)
print(f"check 3  gap above Tc: {above}   {'PASS' if ok3 else 'FAIL'}")

# ---- The bridge: T_BKT against sheet resistance ------------------------------
bkt = np.array([[T_bkt(Ts, Deltas, R, c) for c in (NK, LATTICE)] for R in R_RATIOS])

print("\ncheck 4  limits of the amplitude-to-phase bridge")
print(f"   R_sq/R_Q = {R_RATIOS[0]:.2f}:  T_BKT/Tc = {bkt[0, 1] / Tc:.4f}  "
      "(stiff phase, mean field nearly right)")
print(f"   R_sq/R_Q = {R_RATIOS[-1]:.2f}:  T_BKT/Tc = {bkt[-1, 1] / Tc:.4f}  "
      "(coherence lost far below pairing)")
ok4 = (bkt[0, 1] / Tc > 0.95 and np.all(bkt[:, 1] < bkt[:, 0])
       and np.all(np.diff(bkt[:, 1]) < 0))
print(f"   T_BKT -> Tc at small R, lattice always below bare NK, falls with R   "
      f"{'PASS' if ok4 else 'FAIL'}")
print(f"\n   vortex renormalization costs a further factor "
      f"{np.mean(bkt[:, 1] / bkt[:, 0]):.3f} on average "
      f"(coefficient ratio {LATTICE / NK:.3f})")

np.savez("results_step3.npz", Ts=Ts, Deltas=Deltas, Tc=Tc, U=U, mu=MU,
         R_ratios=R_RATIOS, T_bkt_NK=bkt[:, 0], T_bkt_lattice=bkt[:, 1],
         NK=NK, LATTICE=LATTICE)
print("\nwrote results_step3.npz")
