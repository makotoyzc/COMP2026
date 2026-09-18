"""
Synthetic angle-resolved FMR generator + Smit-Beljers / LOOCV inversion harness.

Purpose: stand up and validate the fitting pipeline BEFORE the real Er:YIG data
arrives. Ground truth is planted, so a correct inversion must recover it.

Convention (verified against Tomczak & Puszkarski, PRB 98, 144415 (2018)):

    (h*f / mu_B)^2  =  g^2 * (1/sin^2 theta) * [f_tt f_pp - f_tp^2]

The left side is a pure constant with NO g in it: for 9.46 GHz it is
(9.46 / 1.39962e-3)^2 Oe^2 = 45.68 kOe^2, which is the paper's 45.6834. g multiplies
the curvature on the right, which is how it becomes a cleanly fitted parameter.
Free energy f is energy density divided by M, i.e. field units (Oe).

Derivatives are exact (sympy, see exact_derivs.py). This matters: nested finite
differences stall the outer fit -- see the note in equilibrium().

Usage:  python sb_synthetic.py 001      # (001) film, 2018-paper-like
        python sb_synthetic.py 111      # (111) garnet film, Er:YIG geometry
"""
import numpy as np
from scipy.optimize import brentq, least_squares
from exact_derivs import FUNCS

MUB_OVER_H = 1.39962e-3          # GHz/Oe
PKEYS = ["Hc1", "Hc2", "Hu2", "Hip", "Meff"]     # free-energy params, in sympy arg order
FIT_KEYS = PKEYS + ["g"]

def _args(th, ph, H, thH, phH, p):
    return (th, ph, H, thH, phH, *[p[k] for k in PKEYS])

# ----------------------------------------------------------------------------
# equilibrium + Smit-Beljers curvature
# ----------------------------------------------------------------------------
def equilibrium(H, thH, phH, p, normal="001"):
    """Newton solve of df/dtheta = df/dphi = 0 using exact derivatives.

    This must converge tightly AND smoothly in the parameters: the outer parameter
    fit finite-differences through this solve. An FD-based inner solver makes the
    outer Jacobian noise and the fit stalls at its initial guess -- that failure is
    real, reproducible, and never mentioned in any of the three papers.
    """
    F = FUNCS[normal]
    best, bf = None, np.inf
    for dth in (0.0, 0.4, -0.4, 0.8):
        t, q = float(np.clip(thH + dth, 1e-3, np.pi - 1e-3)), float(phH)
        for _ in range(80):
            a = _args(t, q, H, thH, phH, p)
            g = np.array([F["ft"](*a), F["fp"](*a)])
            if np.linalg.norm(g) < 1e-10:
                break
            Hs = np.array([[F["ftt"](*a), F["ftp"](*a)], [F["ftp"](*a), F["fpp"](*a)]])
            try:
                step = np.clip(np.linalg.solve(Hs + np.eye(2)*1e-10, g), -0.4, 0.4)
            except np.linalg.LinAlgError:
                break
            t = float(np.clip(t - step[0], 1e-5, np.pi - 1e-5)); q -= float(step[1])
        a = _args(t, q, H, thH, phH, p)
        g = np.array([F["ft"](*a), F["fp"](*a)])
        Hs = np.array([[F["ftt"](*a), F["ftp"](*a)], [F["ftp"](*a), F["fpp"](*a)]])
        # accept only true minima, keep the deepest
        if np.linalg.norm(g) < 1e-7 and Hs[0, 0] > 0 and np.linalg.det(Hs) > 0:
            v = F["f"](*a)
            if v < bf:
                bf, best = v, (t, q)
    return best if best is not None else (np.nan, np.nan)

def curvature(H, thH, phH, p, normal="001"):
    t, q = equilibrium(H, thH, phH, p, normal)
    if not np.isfinite(t):
        return np.nan, np.nan, np.nan
    F = FUNCS[normal]; a = _args(t, q, H, thH, phH, p)
    det = F["ftt"](*a)*F["fpp"](*a) - F["ftp"](*a)**2
    return det / np.sin(t)**2, t, q

def resonance_field(freq_GHz, thH, phH, p, normal="001", Hmax=3e4):
    target = (freq_GHz / MUB_OVER_H)**2
    R = lambda Hx: p["g"]**2 * curvature(Hx, thH, phH, p, normal)[0] - target
    grid = np.linspace(50.0, Hmax, 260)
    pH, pR = grid[0], R(grid[0])
    for Hx in grid[1:]:
        cR = R(Hx)
        if np.isfinite(cR) and np.isfinite(pR) and np.sign(cR) != np.sign(pR):
            return brentq(R, pH, Hx, xtol=1e-6)
        pH, pR = Hx, cR
    return np.nan

# ----------------------------------------------------------------------------
# synthetic datasets
# ----------------------------------------------------------------------------
TRUTH_001 = dict(Hc1=78.07, Hc2=-534.0, Hu2=43.9, Hip=66.3, Meff=4811.0, g=1.985)
TRUTH_111 = dict(Hc1=-61.0, Hc2=5.0,    Hu2=18.0, Hip=6.1,  Meff=826.0,  g=2.005)

def make_dataset(truth, freq_GHz=9.46, normal="001", noise_Oe=2.0, seed=0,
                 dth=7.5, dph=9.0):
    """Out-of-plane + in-plane scans -- the two-geometry layout of the 2018 paper."""
    rng = np.random.default_rng(seed)
    rows  = [(t, np.deg2rad(-45.0)) for t in np.deg2rad(np.arange(0.0, 180.1, dth))]
    rows += [(np.deg2rad(90.0), q)  for q in np.deg2rad(np.arange(0.0, 180.1, dph))]
    out = []
    for thH, phH in rows:
        Hr = resonance_field(freq_GHz, thH, phH, truth, normal)
        if np.isfinite(Hr):
            out.append((thH, phH, Hr + rng.normal(0.0, noise_Oe)))
    return np.array(out)

# ----------------------------------------------------------------------------
# inversion
# ----------------------------------------------------------------------------
def residuals(vec, data, freq_GHz, normal, keys):
    p = dict(zip(keys, vec))
    target = (freq_GHz / MUB_OVER_H)**2
    r = np.empty(len(data))
    for i, (thH, phH, Hr) in enumerate(data):
        c, _, _ = curvature(Hr, thH, phH, p, normal)
        r[i] = (p["g"]**2 * c - target) if np.isfinite(c) else 1e6
    return r / target            # dimensionless relative residual

def fit(data, freq_GHz, x0, normal="001", keys=FIT_KEYS):
    scale = np.array([100., 500., 50., 50., 1000., 1.])[:len(x0)]
    r = least_squares(residuals, x0, args=(data, freq_GHz, normal, keys),
                      method="trf", x_scale=scale, diff_step=1e-6,
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=6000)
    return dict(zip(keys, r.x)), r

def rms(vec, data, freq_GHz, normal="001", keys=FIT_KEYS):
    return float(np.sqrt(np.mean(residuals(np.asarray(vec, float),
                                           data, freq_GHz, normal, keys)**2)))

def loocv(data, freq_GHz, x0, normal="001", keys=FIT_KEYS):
    """(mean train RMS, mean held-out RMS). Held-out is the model-selection number."""
    tr, te = [], []
    for i in range(len(data)):
        m = np.ones(len(data), bool); m[i] = False
        _, r = fit(data[m], freq_GHz, x0, normal, keys)
        tr.append(rms(r.x, data[m],  freq_GHz, normal, keys))
        te.append(rms(r.x, data[~m], freq_GHz, normal, keys))
    return float(np.mean(tr)), float(np.mean(te))

if __name__ == "__main__":
    import sys, time
    normal = sys.argv[1] if len(sys.argv) > 1 else "001"
    truth  = TRUTH_001 if normal == "001" else TRUTH_111
    freq   = 9.46 if normal == "001" else 8.0
    tv     = np.array([truth[k] for k in FIT_KEYS])

    t0 = time.time()
    data = make_dataset(truth, freq, normal, noise_Oe=2.0, seed=1)
    print(f"--- synthetic ({normal}-normal, {freq} GHz), N={len(data)}, "
          f"Hr {data[:,2].min():.0f}-{data[:,2].max():.0f} Oe, {time.time()-t0:.1f}s ---")

    x0 = tv*0.7; x0[-1] = 2.0
    p, r = fit(data, freq, x0, normal)
    rv = np.array([p[k] for k in FIT_KEYS])
    print(f"\nRMS  truth={rms(tv,data,freq,normal):.3e}  fit={rms(rv,data,freq,normal):.3e}")
    print(f"\n{'param':6s} {'truth':>10s} {'recovered':>12s} {'err %':>8s}")
    for k, a, b in zip(FIT_KEYS, tv, rv):
        print(f"{k:6s} {a:10.3f} {b:12.3f} {100*(b-a)/a:8.2f}")

    np.savetxt(f"synthetic_{normal}.csv",
               np.column_stack([np.rad2deg(data[:,0]), np.rad2deg(data[:,1]), data[:,2]]),
               delimiter=",", header="thetaH_deg,phiH_deg,Hr_Oe", comments="")
    print(f"\nwrote synthetic_{normal}.csv   ({time.time()-t0:.1f}s total)")
