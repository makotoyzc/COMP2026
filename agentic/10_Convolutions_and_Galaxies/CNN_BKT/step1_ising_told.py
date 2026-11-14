"""Step 1: Ising, told the answer. Label = (T < T_c). Prints Check 1.

One network per L (seed 0), trained on chains 0-3, tested on chains 4-5,
with the global spin flip as augmentation. Baseline: logistic regression on
|m| alone.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import train
from data import ising

LS = (16, 32)
TC = ising.TC


def main():
    res, ok = {}, []
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for L in LS:
        d = data.load("ising", L)
        sp = train.split(d)
        tr, te = sp["train"], sp["test"]
        y_tr, y_te = tr["T"] < TC, te["T"] < TC
        r = train.fit(train.ising_input(tr["x"]), y_tr, train.ising_input(te["x"]), y_te,
                      seed=0, augment=train.flip)
        Ts = d["Ts"]
        p_T = np.array([r["p"][te["T"] == T].mean() for T in Ts])
        acc_m = train.logistic(tr["absm"], y_tr, te["absm"], y_te)
        res[L] = dict(acc=r["acc"], acc_m=acc_m, Tstar=train.crossing(Ts, p_T), p_T=p_T,
                      r2=train.binned_r2(te["absm"], r["p"]), p=r["p"], absm=te["absm"], T=te["T"])
        print(f"L={L}: test acc {r['acc']:.4f}, |m| baseline {acc_m:.4f}, "
              f"T* = {res[L]['Tstar']:.4f}, output-vs-|m| R^2 = {res[L]['r2']:.4f}")
        ax[0].plot(Ts, p_T, "o-", ms=4, label=f"L = {L}, T* = {res[L]['Tstar']:.3f}")

    t16, t32 = res[16]["Tstar"], res[32]["Tstar"]
    ok.append(abs(t32 - TC) < 0.05)
    print(f"Check 1.1  T*(32) = {t32:.4f}, |T* - Tc| = {abs(t32 - TC):.4f}  (< 0.05)")
    ok.append(abs(t16 - TC) > abs(t32 - TC))
    print(f"Check 1.2  |T*(16) - Tc| = {abs(t16 - TC):.4f} > |T*(32) - Tc| = {abs(t32 - TC):.4f}")
    r32 = res[32]
    ok.append(abs(r32["acc"] - r32["acc_m"]) <= 0.01 and r32["r2"] > 0.95)
    print(f"Check 1.3  L=32: CNN acc - |m| acc = {r32['acc'] - r32['acc_m']:+.4f} (within 0.01); "
          f"output-vs-|m| R^2 = {r32['r2']:.4f} (> 0.95)")
    print("Check 1:", "PASS" if all(ok) else "FAIL", ok)

    ax[0].axvline(TC, color="gray", ls=":")
    ax[0].axhline(0.5, color="gray", lw=0.5)
    ax[0].set(xlabel="T", ylabel="mean output p(ordered), test chains",
              title="Ising, told T_c")
    ax[0].legend()
    sc = ax[1].scatter(r32["absm"], r32["p"], c=r32["T"], s=3, cmap="coolwarm")
    fig.colorbar(sc, ax=ax[1], label="T")
    ax[1].set(xlabel="|m| of snapshot", ylabel="network output",
              title=f"L = 32: output vs |m|, R^2 = {r32['r2']:.3f}")
    fig.tight_layout()
    fig.savefig(data.HERE / "figures" / "step1_check.png", dpi=120)

    np.savez(data.HERE / "results_step1.npz", Ls=LS, Ts=data.GRIDS["ising"],
             **{f"{k}_{L}": res[L][k] for L in LS for k in ("acc", "acc_m", "Tstar", "p_T", "r2")})


if __name__ == "__main__":
    main()
