"""Step 4: a critical point and a critical line, measured side by side.

Ising and XY on the same lattice, with exact answers to check against.

  0. Swendsen-Wang and Metropolis agree for Ising (one is only faster).
  1. Ising E/N matches Onsager away from Tc.
  2. Ising |m| matches Onsager below Tc.
  3. Ising at Tc: |m| ~ L^(-beta/nu) with beta/nu = 1/8. A critical POINT --
     this power law holds at Tc and nowhere else.
  4. XY runs are equilibrated at the largest L and the slowest T.
  5. XY at every T below T_BKT: <m^2> ~ L^(-eta(T)). A critical LINE. And the
     exponent is not free -- step 2's stiffness predicts it,
     eta(T) = T / (2 pi Y(T)), with no fitted parameters.
  6. eta = 1/4 at T_BKT, universal and exact.
  7. The Ising specific-heat peak sharpens with L; the XY one does not, because
     it is not at a transition at all.

Run:  ~/anaconda3/bin/python3 checks_step4.py   (about 12 min on 18 cores)
Writes results_step4.npz. Make the figures with plot_step4.py.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from ising import TC, m_exact, e_exact, run as ising_run
from jump import T_BKT
from scaling import xy_measure, ising_measure, power_fit

XY_LS = [8, 16, 32, 64]
XY_TS = [0.70, 0.80, 0.88, 0.90, 1.00]          # a subset of step 2's grid
XY_HEAT_LS, XY_HEAT_TS = [16, 32], [0.60, 0.70, 0.80, 0.90, 0.95, 1.00,
                                    1.05, 1.10, 1.20, 1.40, 1.60]
XY_THERM, XY_MEAS = 20000, 200000

IS_LS = [8, 16, 32, 64, 128]
IS_TS = [1.6, 1.8, 2.0, 2.1, 2.2, TC, 2.35, 2.45, 2.6, 2.8, 3.0, 3.2]
IS_THERM, IS_MEAS = 5000, 100000

SNAP_XY, SNAP_IS = (64, 1.10), (64, 1.80)
EQ_XY = (64, 0.90)
WORKERS = 14

if __name__ == "__main__":
    xy_jobs = sorted({(L, T) for L in XY_LS for T in XY_TS} |
                     {(L, T) for L in XY_HEAT_LS for T in XY_HEAT_TS} |
                     {SNAP_XY}, key=lambda j: -j[0])
    is_jobs = sorted({(L, T) for L in IS_LS for T in IS_TS}, key=lambda j: -j[0])
    n = len(xy_jobs) + len(is_jobs) + 1
    seeds = np.random.SeedSequence(21).spawn(n)
    print(f"{n} runs on {WORKERS} workers")

    res, done = {}, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        f = [pool.submit(xy_measure, L, T, XY_THERM, XY_MEAS, s)
             for (L, T), s in zip(xy_jobs, seeds)]
        f += [pool.submit(ising_measure, L, T, IS_THERM, IS_MEAS, s)
              for (L, T), s in zip(is_jobs, seeds[len(xy_jobs):])]
        # the XY equilibration check: same point, opposite start
        f.append(pool.submit(xy_measure, *EQ_XY, XY_THERM, XY_MEAS, seeds[-1],
                             "random"))
        for fut in as_completed(f):
            r = fut.result()
            done += 1
            key = (r["model"], r["L"], r["T"])
            if r.get("start") == "random":
                key += ("random",)
            res[key] = r
            if done % 10 == 0:
                print(f"  [{done:3d}/{n}]", flush=True)

    xy = lambda L, T: res[("xy", L, T)]
    isg = lambda L, T: res[("ising", L, T)]

    # ---- Check 0: the two Ising algorithms agree --------------------------------
    a = ising_run(32, 2.0, 2000, 20000, 99, algo="metropolis")
    mh = np.abs(a["rec"]["m"]).mean()
    b = isg(32, 2.0)
    ok0 = abs(mh - b["m"]) < 4 * b["dm"]
    print(f"\ncheck 0  Ising L=32 T=2.0: |m| = {mh:.4f} (Metropolis) vs "
          f"{b['m']:.4f} +/- {b['dm']:.4f} (Swendsen-Wang)   {'PASS' if ok0 else 'FAIL'}")

    # ---- Checks 1 and 2: Onsager ------------------------------------------------
    print("\ncheck 1/2  Ising against Onsager at L = 128")
    ok12 = True
    for T in (1.6, 1.8, 2.0, 2.6, 3.0, 3.2):
        r = isg(128, T)
        de, dm = abs(r["E"] - e_exact(T)), abs(r["m"] - m_exact(T))
        good = de < 4 * r["dE"] + 2e-3 and (T > TC or dm < 4 * r["dm"] + 2e-3)
        ok12 &= good
        print(f"   T = {T:.1f}:  E/N = {r['E']:+.4f} vs {float(e_exact(T)):+.4f}"
              f"   |m| = {r['m']:.4f} vs {float(m_exact(T)):.4f}"
              f"   {'ok' if good else 'OFF'}")
    print(f"   {'PASS' if ok12 else 'FAIL'}")

    # ---- Check 3: Ising at Tc, the critical point -------------------------------
    mTc = np.array([[isg(L, TC)["m"], isg(L, TC)["dm"]] for L in IS_LS])
    b_nu, db = power_fit(IS_LS, mTc[:, 0], mTc[:, 1])
    ok3 = abs(b_nu - 0.125) < 0.02
    print(f"\ncheck 3  Ising |m| ~ L^-x at Tc:  x = {b_nu:.4f} +/- {db:.4f}"
          f"   vs beta/nu = 0.125   {'PASS' if ok3 else 'FAIL'}")
    print(f"   E/N at Tc: " + ", ".join(f"L={L}: {isg(L, TC)['E']:.4f}" for L in IS_LS)
          + f"   -> exact -{np.sqrt(2):.4f}")

    # ---- Check 4: XY equilibration ----------------------------------------------
    p, q = res[("xy",) + EQ_XY], res[("xy",) + EQ_XY + ("random",)]
    sig = abs(p["m2"] - q["m2"]) / np.hypot(p["dm2"], q["dm2"])
    ok4 = sig < 3
    print(f"\ncheck 4  XY equilibration at L = {EQ_XY[0]}, T = {EQ_XY[1]}:  "
          f"<m^2> = {p['m2']:.5f} +/- {p['dm2']:.5f} vs {q['m2']:.5f} +/- {q['dm2']:.5f}"
          f"   {sig:.1f} sigma   {'PASS' if ok4 else 'FAIL'}")

    # ---- Checks 5 and 6: the XY critical line ------------------------------------
    print("\ncheck 5/6  XY <m^2> ~ L^-eta, against eta = T / (2 pi Y) from the stiffness")
    eta = []
    for T in XY_TS:
        m2 = np.array([[xy(L, T)["m2"], xy(L, T)["dm2"]] for L in XY_LS])
        e, de = power_fit(XY_LS, m2[:, 0], m2[:, 1])
        Y, dY = xy(XY_LS[-1], T)["Y"], xy(XY_LS[-1], T)["dY"]
        pred = T / (2 * np.pi * Y)
        eta.append((T, e, de, pred))
        print(f"   T = {T:.2f}:  eta = {e:.4f} +/- {de:.4f}   predicted {pred:.4f}"
              f"   Y(L=64) = {Y:.4f}")
    eta = np.array(eta)
    below = eta[:, 0] < T_BKT
    ok5 = np.all(np.abs(eta[below, 1] - eta[below, 3]) < 4 * eta[below, 2] + 0.02)
    at = np.argmin(np.abs(eta[:, 0] - T_BKT))
    ok6 = abs(eta[at, 1] - 0.25) < 0.04
    print(f"   below T_BKT, measured matches predicted   {'PASS' if ok5 else 'FAIL'}")
    print(f"   at T = {eta[at, 0]:.2f} (nearest T_BKT = {T_BKT}): eta = {eta[at, 1]:.4f}"
          f"   vs 1/4   {'PASS' if ok6 else 'FAIL'}")

    # ---- Check 7: one peak is a transition, the other is not --------------------
    isC = np.array([max(isg(L, T)["C"] for T in IS_TS) for L in IS_LS])
    xyC = np.array([max(xy(L, T)["C"] for T in XY_HEAT_TS) for L in XY_HEAT_LS])
    ok7 = isC[-1] / isC[0] > 1.5 and abs(xyC[1] / xyC[0] - 1) < 0.1
    print(f"\ncheck 7  peak specific heat")
    print("   Ising: " + ", ".join(f"L={L}: {c:.3f}" for L, c in zip(IS_LS, isC))
          + f"   grows by {isC[-1] / isC[0]:.2f}x")
    print("   XY:    " + ", ".join(f"L={L}: {c:.3f}" for L, c in zip(XY_HEAT_LS, xyC))
          + f"   grows by {xyC[1] / xyC[0]:.2f}x")
    print(f"   {'PASS' if ok7 else 'FAIL'}")

    np.savez("results_step4.npz", XY_LS=XY_LS, XY_TS=XY_TS, IS_LS=IS_LS, IS_TS=IS_TS,
             XY_HEAT_LS=XY_HEAT_LS, XY_HEAT_TS=XY_HEAT_TS, eta=eta, beta_nu=[b_nu, db],
             xy_m2=[[xy(L, T)["m2"] for T in XY_TS] for L in XY_LS],
             xy_dm2=[[xy(L, T)["dm2"] for T in XY_TS] for L in XY_LS],
             xy_C=[[xy(L, T)["C"] for T in XY_HEAT_TS] for L in XY_HEAT_LS],
             xy_dC=[[xy(L, T)["dC"] for T in XY_HEAT_TS] for L in XY_HEAT_LS],
             is_m=[[isg(L, T)["m"] for T in IS_TS] for L in IS_LS],
             is_dm=[[isg(L, T)["dm"] for T in IS_TS] for L in IS_LS],
             is_C=[[isg(L, T)["C"] for T in IS_TS] for L in IS_LS],
             is_dC=[[isg(L, T)["dC"] for T in IS_TS] for L in IS_LS],
             is_mTc=mTc, TC=TC, T_BKT=T_BKT,
             snap_xy_cold=xy(64, 0.70)["config"], snap_xy_mid=xy(64, 0.90)["config"],
             snap_xy_hot=xy(*SNAP_XY)["config"],
             snap_is_cold=isg(*SNAP_IS)["config"], snap_is_mid=isg(64, TC)["config"],
             snap_is_hot=isg(64, 2.8)["config"])
    print("\nwrote results_step4.npz")
