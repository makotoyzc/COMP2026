"""Step 2: Ising, not told (learning by confusion). Prints Check 2.

For every trial cut T' on the grid: label each snapshot by (T < T'), train a
fresh network, record test accuracy. Accuracy vs T' is W-shaped; the middle
peak is where the data really changes (van Nieuwenburg, Liu and Huber 2017).
L = 32, 3 training seeds, 29 cuts each: 87 trainings, run on GPU + CPU
workers together (train.run_parallel), about 6 minutes.

    python step2_ising_confusion.py          # trains if results_step2.npz is missing, then prints the check
"""
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import data
import train
from data import ising

L = 32
SEEDS = (0, 1, 2)
OUT = data.HERE / "results_step2.npz"


@lru_cache
def inputs():
    """Loaded once per worker process."""
    sp = train.split(data.load("ising", L))
    tr, te = sp["train"], sp["test"]
    return train.ising_input(tr["x"]), tr["T"], train.ising_input(te["x"]), te["T"]


def one_cut(job):
    seed, cut = job
    x_tr, T_tr, x_te, T_te = inputs()
    return train.fit(x_tr, T_tr < cut, x_te, T_te < cut, seed=seed, augment=train.flip)["acc"]


def main():
    cuts = data.GRIDS["ising"]
    if not OUT.exists():
        jobs = [(seed, cut) for seed in SEEDS for cut in cuts]
        acc = np.array(train.run_parallel(one_cut, jobs)).reshape(len(SEEDS), len(cuts))
        np.savez(OUT, cuts=cuts, L=L, **{f"acc_{seed}": a for seed, a in zip(SEEDS, acc)})
    saved = dict(np.load(OUT))

    tc = ising.TC
    ok, peaks = [], []
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for seed in SEEDS:
        acc = saved[f"acc_{seed}"]
        peak, sub, prom = train.middle_peak(cuts, acc)
        peaks.append(peak)
        i = int(np.argmin(np.abs(cuts - 0.025 - peak)))
        w = acc[0] > 0.95 and acc[-1] > 0.95 and acc[:i].min() < acc[i] and acc[i + 1:].min() < acc[i]
        ok.append(w)
        print(f"seed {seed}: ends {acc[0]:.3f} / {acc[-1]:.3f}, dips {acc[:i].min():.3f} "
              f"(T' = {cuts[acc[:i].argmin()]:.2f}) / {acc[i + 1:].min():.3f} "
              f"(T' = {cuts[i + 1 + acc[i + 1:].argmin()]:.2f}), "
              f"peak {acc[i]:.4f} at T' = {cuts[i]:.2f}, boundary T = {peak:.3f} "
              f"(parabola {sub:.3f}), prominence {prom:.3f}")
        ax.plot(cuts - 0.025, acc, "o-", ms=4, label=f"seed {seed}: peak at {peak:.3f}")
    peaks = np.array(peaks)
    print(f"Check 2.1  W shape (ends > 0.95, a dip on each side of the peak), all seeds: {all(ok)}")
    ok2 = abs(peaks[0] - tc) <= 0.05 + 1e-9
    print(f"Check 2.2  seed-0 peak boundary {peaks[0]:.3f}, |peak - Tc| = {abs(peaks[0] - tc):.3f}  (<= 0.05)")
    ok3 = peaks.max() - peaks.min() <= 0.05 + 1e-9
    print(f"Check 2.3  peaks over seeds {peaks}, spread {peaks.max() - peaks.min():.2f}  (<= 0.05)")
    print("Check 2:", "PASS" if all(ok) and ok2 and ok3 else "FAIL")

    ax.axvline(tc, color="gray", ls=":", label="exact T_c")
    ax.set(xlabel="boundary of trial labels, T' - 0.025", ylabel="test accuracy", title=f"Ising confusion, L = {L}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(data.HERE / "figures" / "step2_check.png", dpi=120)


if __name__ == "__main__":
    main()
