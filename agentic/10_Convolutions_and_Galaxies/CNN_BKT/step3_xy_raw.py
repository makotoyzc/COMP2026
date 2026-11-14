"""Step 3: XY from raw spins, input (cos theta, sin theta), random global rotation
as augmentation. Prints Check 3 from whatever results exist so far.

    python step3_xy_raw.py told                    # 3a: label T < 0.893, L = 16, 32, 64
    python step3_xy_raw.py confusion 5x5 16 32     # 3b: confusion, 3 seeds, given L
    python step3_xy_raw.py confusion 9x9 16 32     # 3c: same with the four-layer net
    python step3_xy_raw.py check                   # only print the check, redraw figures

Each run adds its keys to results_step3.npz, so the pieces can run separately.
Cost on this Mac (GPU, one training): 5x5 net 2 / 4 / 17 s and 9x9 net
3 / 12 / 45 s at L = 16 / 32 / 64. A confusion set is 21 cuts x 3 seeds.
"""
import sys
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import train

T_BKT = 0.893
LS = (16, 32, 64)
SEEDS = (0, 1, 2)
NETS = {"5x5": ((8, 3), (16, 3)), "9x9": ((8, 3), (16, 3), (16, 3), (16, 3))}
FEATS = {"absm": "|m|", "e": "energy per site", "nv": "vortex density"}
OUT = data.HERE / "results_step3.npz"


def load_results():
    return dict(np.load(OUT)) if OUT.exists() else {}


def save_results(new):
    res = load_results()
    res.update(new)
    np.savez(OUT, **res)


@lru_cache
def inputs(L):
    """Loaded once per worker process."""
    sp = train.split(data.load("xy", L))
    tr, te = sp["train"], sp["test"]
    return train.xy_input(tr["x"]), tr["T"], train.xy_input(te["x"]), te["T"]


def one_cut(job):
    net, L, seed, cut = job
    x_tr, T_tr, x_te, T_te = inputs(L)
    return train.fit(x_tr, T_tr < cut, x_te, T_te < cut, seed=seed,
                     layers=NETS[net], augment=train.rotate)["acc"]


def told():
    new = {}
    for L in LS:
        sp = train.split(data.load("xy", L))
        tr, te = sp["train"], sp["test"]
        y_tr, y_te = tr["T"] < T_BKT, te["T"] < T_BKT
        r = train.fit(train.xy_input(tr["x"]), y_tr, train.xy_input(te["x"]), y_te,
                      seed=0, augment=train.rotate)
        Ts = data.GRIDS["xy"]
        p_T = np.array([r["p"][te["T"] == T].mean() for T in Ts])
        new.update({f"told_acc_{L}": r["acc"], f"told_pT_{L}": p_T, f"told_p_{L}": r["p"],
                    f"told_Tstar_{L}": train.crossing(Ts, p_T)})
        for k in FEATS:
            new[f"base_{k}_{L}"] = train.logistic(tr[k], y_tr, te[k], y_te)
            new[f"r2_{k}_{L}"] = train.binned_r2(te[k], r["p"])
        print(f"told L={L}: acc {r['acc']:.4f}, T* = {new[f'told_Tstar_{L}']:.4f}", flush=True)
    save_results(new)


def confusion(net, Ls):
    cuts = data.GRIDS["xy"]
    for L in Ls:
        jobs = [(net, L, seed, cut) for seed in SEEDS for cut in cuts]
        acc = np.array(train.run_parallel(one_cut, jobs)).reshape(len(SEEDS), len(cuts))
        save_results({f"conf_{net}_{L}": acc})
        print(f"confusion {net} L={L} done", flush=True)


def check():
    res = load_results()
    cuts = data.GRIDS["xy"]
    Ts = data.GRIDS["xy"]
    figs = data.HERE / "figures"

    # 3a and Check 3.3
    if "told_acc_16" in res:
        print("3a, told (label T < 0.893), seed 0:")
        print("   L   CNN acc   T*      | baseline acc: |m|   energy  vortex | R^2 of output vs: |m|   energy  vortex")
        for L in LS:
            b = [res[f"base_{k}_{L}"] for k in FEATS]
            r2 = [res[f"r2_{k}_{L}"] for k in FEATS]
            print(f"  {L:2d}   {res[f'told_acc_{L}']:.4f}   {res[f'told_Tstar_{L}']:.4f}  |"
                  f"               {b[0]:.4f}  {b[1]:.4f}  {b[2]:.4f} |"
                  f"                    {r2[0]:.3f}   {r2[1]:.3f}   {r2[2]:.3f}")
        best = {L: max(FEATS, key=lambda k: res[f"r2_{k}_{L}"]) for L in LS}
        print("Check 3.3  best-matching single number (highest R^2):",
              ", ".join(f"L={L}: {FEATS[k]}" for L, k in best.items()))

        fig, ax = plt.subplots(1, 4, figsize=(17, 4))
        for L in LS:
            ax[0].plot(Ts, res[f"told_pT_{L}"], "o-", ms=4,
                       label=f"L = {L}, T* = {res[f'told_Tstar_{L}']:.3f}")
        ax[0].axvline(T_BKT, color="gray", ls=":")
        ax[0].axhline(0.5, color="gray", lw=0.5)
        ax[0].set(xlabel="T", ylabel="mean output, test chains", title="XY raw spins, told T_BKT")
        ax[0].legend(fontsize=8)
        L = 32
        te = train.split(data.load("xy", L))["test"]
        for a, k in zip(ax[1:], FEATS):
            sc = a.scatter(te[k], res[f"told_p_{L}"], c=te["T"], s=2, cmap="coolwarm")
            a.set(xlabel=FEATS[k] + " of snapshot", ylabel="network output",
                  title=f"L = {L}: R^2 = {res[f'r2_{k}_{L}']:.3f}")
        fig.colorbar(sc, ax=ax[3], label="T")
        fig.tight_layout()
        fig.savefig(figs / "step3_told.png", dpi=110)

    # 3b, 3c: Checks 3.1, 3.2, 3.4
    have = [(net, L) for net in NETS for L in LS if f"conf_{net}_{L}" in res]
    if not have:
        return
    peaks = {}
    print("Confusion peaks (label boundary T' - 0.025; parabola in brackets):")
    for net, L in have:
        acc = res[f"conf_{net}_{L}"]
        pk = np.array([train.middle_peak(cuts, a) for a in acc])
        peaks[net, L] = pk[:, 0]
        spread = pk[:, 0].max() - pk[:, 0].min()
        print(f"  {net} L={L:2d}: seeds {pk[:, 0]} [{', '.join(f'{x:.3f}' for x in pk[:, 1])}], "
              f"prominence {pk[:, 2].round(3)}, spread {spread:.2f} -> 3.1 "
              f"{'ok' if spread <= 0.05 + 1e-9 else 'FAIL'}")
        print("        mean acc vs cut:", " ".join(f"{x:.3f}" for x in acc.mean(0)))
    print("Check 3.2  5x5 peak vs prediction (0.893 = bound/free; 1.05-1.1 = vortex count / C peak):")
    for L in LS:
        if ("5x5", L) in peaks:
            m = np.median(peaks["5x5", L])
            print(f"  L={L}: {m:.3f}, {m - T_BKT:+.3f} from T_BKT")
    if all(("9x9", L) in peaks and ("5x5", L) in peaks for L in LS[:2]):
        print("Check 3.4  9x9 vs 5x5 peak:",
              ", ".join(f"L={L}: {np.median(peaks['9x9', L]):.3f} vs {np.median(peaks['5x5', L]):.3f}"
                        for L in LS if ("9x9", L) in peaks))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for j, net in enumerate(NETS):
        for L in LS:
            if (net, L) in peaks:
                acc = res[f"conf_{net}_{L}"]
                ax[j].errorbar(cuts - 0.025, acc.mean(0), acc.std(0), fmt="o-", ms=3,
                               capsize=2, label=f"L = {L}")
        ax[j].axvline(T_BKT, color="gray", ls=":")
        ax[j].axvspan(1.05, 1.1, color="orange", alpha=0.15)
        ax[j].set(xlabel="label boundary T", ylabel="test accuracy (3 seeds)",
                  title=f"XY raw spins, confusion, {net} window")
        ax[j].legend(fontsize=8)
    for net, mk in (("5x5", "o-"), ("9x9", "s--")):
        Lh = [L for L in LS if (net, L) in peaks]
        if Lh:
            ax[2].errorbar(Lh, [np.median(peaks[net, L]) for L in Lh],
                           [np.ptp(peaks[net, L]) / 2 for L in Lh], fmt=mk, capsize=3,
                           label=f"{net} window")
    ax[2].axhline(T_BKT, color="gray", ls=":", label="T_BKT")
    ax[2].axhspan(1.05, 1.1, color="orange", alpha=0.15, label="specific-heat bump")
    ax[2].set(xscale="log", xticks=LS, xticklabels=LS, xlabel="L",
              ylabel="confusion peak T", title="peak vs L and window")
    ax[2].minorticks_off()
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "step3_confusion.png", dpi=110)


if __name__ == "__main__":
    if sys.argv[1] == "told":
        told()
    elif sys.argv[1] == "confusion":
        confusion(sys.argv[2], [int(x) for x in sys.argv[3:]])
    check()
