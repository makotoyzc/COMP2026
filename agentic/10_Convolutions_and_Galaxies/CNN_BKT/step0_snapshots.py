"""Step 0: make Ising and XY snapshots for L = 16, 32 and run Check 0.

    python step0_snapshots.py          # make missing data, then check
    python step0_snapshots.py --remake # regenerate everything
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
from data import ising

LS = (16, 32)
GAP = {"ising": 50, "xy": 20}  # XY sweeps include a cluster step
N_THERM = {"ising": 2000, "xy": 2000}
WEEK7 = data.HERE.parent.parent / "07_Stat_Mech_and_Ising" / "XY_BKT" / "results_step2.npz"


def lag1(x):
    """Lag-1 autocorrelation of each chain's series, averaged over chains.
    x has shape (nT, n_chain, n_snap); returns one number per T."""
    a, b = x[..., :-1], x[..., 1:]
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    r = (a * b).mean(-1) / np.sqrt((a**2).mean(-1) * (b**2).mean(-1))
    return r.mean(1)


def chain_mean_err(x):
    """Mean over everything; error from the spread of the chain means."""
    cm = x.mean(-1)
    return cm.mean(1), cm.std(1, ddof=1) / np.sqrt(cm.shape[1])


def extra_nv(args):
    L, iT, c = args
    return data.xy_chain(L, iT, c, GAP["xy"], N_THERM["xy"])["nv"].mean()


def main():
    for model in ("ising", "xy"):
        for L in LS:
            f = data.DATA / f"{model}_L{L}.npz"
            if "--remake" in sys.argv or not f.exists():
                data.make(model, L, GAP[model], N_THERM[model])
                print("made", f.name)

    ok = []
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # 1, 2: Ising against Onsager at L = 32
    d = data.load("ising", 32)
    Ts = d["Ts"]
    e, de = chain_mean_err(d["e"])
    m, dm = chain_mean_err(d["absm"])
    far = np.abs(Ts - ising.TC) > 0.2
    err_e = np.abs(e - ising.e_exact(Ts))[far].max()
    low = Ts <= 2.1
    err_m = np.abs(m - ising.m_exact(Ts))[low].max()
    ok += [err_e < 0.01, err_m < 0.02]
    print(f"Check 0.1  Ising e vs Onsager, |T-Tc|>0.2, L=32: max |diff| = {err_e:.4f}  (< 0.01)")
    print(f"Check 0.2  Ising |m| vs Onsager, T<=2.1, L=32:   max |diff| = {err_m:.4f}  (< 0.02)")
    Tf = np.linspace(1.6, 3.0, 300)
    ax[0, 0].plot(Tf, ising.e_exact(Tf), "k-", label="Onsager")
    ax[0, 0].errorbar(Ts, e, de, fmt="o", ms=4, label="L = 32")
    ax[0, 0].set(xlabel="T", ylabel="energy per site", title="Ising energy")
    ax[0, 1].plot(Tf, ising.m_exact(Tf), "k-", label="Onsager")
    for L in LS:
        mm, dmm = chain_mean_err(data.load("ising", L)["absm"])
        ax[0, 1].errorbar(Ts, mm, dmm, fmt="o", ms=4, label=f"L = {L}")
    ax[0, 1].set(xlabel="T", ylabel="<|m|>", title="Ising magnetization")
    for a in ax[0]:
        a.axvline(ising.TC, color="gray", ls=":")
        a.legend()

    # 3: XY vortex density against week 7
    ref = np.load(WEEK7)
    print("Check 0.3  XY vortex density vs XY_BKT results_step2.npz:")
    worst, at = 0.0, None
    for L in LS:
        d = data.load("xy", L)
        nv, dnv = chain_mean_err(d["nv"])
        iL = list(ref["Ls"]).index(L)
        for iT, T in enumerate(d["Ts"]):
            hit = np.isclose(ref["Ts"], T)
            if not hit.any():
                continue
            r, dr = ref["n_vort"][iL][hit][0], ref["dn_vort"][iL][hit][0]
            z = (nv[iT] - r) / np.hypot(dnv[iT], dr)
            if abs(z) > worst:
                worst, at = abs(z), (L, iT, r, dr)
            print(f"    L={L:2d} T={T:.2f}  ours {nv[iT]:.5f} +- {dnv[iT]:.5f}   "
                  f"week7 {r:.5f} +- {dr:.5f}   z = {z:+.1f}")
        ax[1, 0].errorbar(d["Ts"], nv, dnv, fmt="o", ms=4, label=f"L = {L}, this step")
        ax[1, 0].errorbar(ref["Ts"], ref["n_vort"][iL], ref["dn_vort"][iL], fmt="k+",
                          ms=8, label=f"L = {L}, week 7" if L == LS[0] else None)
    print(f"    worst |z| = {worst:.1f}  (< 2)")
    if worst >= 2:
        # Six chains give a rough error bar. Rerun the worst point with 48 new
        # chains (indices 6-53, never used for training) to tell a sampler bug
        # from a fluctuation.
        L, iT, r, dr = at
        with ProcessPoolExecutor(16) as ex:
            m = np.array(list(ex.map(extra_nv, [(L, iT, c) for c in range(6, 54)])))
        mm, dmm = m.mean(), m.std(ddof=1) / np.sqrt(len(m))
        worst = abs(mm - r) / np.hypot(dmm, dr)
        print(f"    0.3b rerun L={L} T={data.GRIDS['xy'][iT]:.2f} with 48 new chains: "
              f"{mm:.5f} +- {dmm:.5f}, z = {(mm - r) / np.hypot(dmm, dr):+.1f}  (< 2)")
    ok.append(worst < 2)
    ax[1, 0].axvline(0.893, color="gray", ls=":")
    ax[1, 0].set(xlabel="T", ylabel="vortex density", title="XY vortices", yscale="log")
    ax[1, 0].legend(fontsize=8)

    # 4: independence of consecutive snapshots
    print("Check 0.4  lag-1 correlation of |m| between saved snapshots (all < 0.2):")
    for model, mk in (("ising", "o-"), ("xy", "s-")):
        for L in LS:
            d = data.load(model, L)
            r = lag1(d["absm"])
            ok.append(r.max() < 0.2)
            print(f"    {model:5s} L={L:2d}: max {r.max():.3f} at T = {d['Ts'][r.argmax()]:.2f}")
            ax[1, 1].plot(d["Ts"], r, mk, ms=4, label=f"{model} L = {L}")
    ax[1, 1].axhline(0.2, color="r", ls="--")
    ax[1, 1].set(xlabel="T", ylabel="lag-1 correlation of |m|",
                 title="snapshot independence (gap: Ising 50, XY 20)")
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    (data.HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(data.HERE / "figures" / "step0_check.png", dpi=120)
    print("Check 0:", "PASS" if all(ok) else "FAIL", ok)


if __name__ == "__main__":
    main()
