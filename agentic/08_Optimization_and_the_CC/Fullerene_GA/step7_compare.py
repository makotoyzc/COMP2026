"""Step 7: the three searches for C60 compared at equal cost (energy+force calls).

Left:  probability of having found C60 (within 1e-3 eV/atom of E_ref) within a budget.
       Multistart and annealing are repeatable: B calls buy B / cost independent runs, so
       P = 1 - (1 - q)^(B / cost) with q the measured success rate per run. For the GA, P is
       the fraction of ecologies that had found C60 by B calls.
Right: how close each method gets: best energy above E_ref against calls spent.

Usage:  python3 step7_compare.py     (needs step4_results.npz, step5_results.npz, ga_runs/)
        -> step7_compare.png
"""
import glob

import numpy as np
import matplotlib.pyplot as plt

s4 = np.load("step4_results.npz", allow_pickle=True)
s5 = np.load("step5_results.npz", allow_pickle=True)
E_ref = float(s4["E_ref"])
ga_files = sorted(f for f in glob.glob("ga_runs/C60_mu0.0_seed*.npz") if "aligned" not in f)
ga = [np.load(f) for f in ga_files]
B = np.logspace(4, 7.5, 300)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# ---- multistart: 720 starts in order
ms_E, ms_c = s4["Eb60"], s4["nfev60"]
q_ms = np.mean(ms_E < E_ref + 1e-3)
ax1.plot(B, 1 - (1 - q_ms) ** (B / ms_c.mean()), "C0", label=f"random multistart (q = {q_ms:.0%} per start)")
ax2.plot(np.cumsum(ms_c), np.minimum.accumulate(ms_E) - E_ref, "C0", label="random multistart, best so far")
print(f"multistart: {len(ms_E)} starts, {ms_c.sum()} calls, success rate {q_ms:.3%},"
      f" best {ms_E.min() - E_ref:+.4f} above E_ref")

# ---- annealing: one point per schedule
m60 = s5["N"] == 60
for k, L in enumerate(sorted(set(s5["steps"][m60]))):
    sel = m60 & (s5["steps"] == L)
    E, c = s5["Eb"][sel], s5["calls"][sel].mean()
    q = np.mean(E < E_ref + 1e-3)
    ax1.plot(B, 1 - (1 - q) ** (B / c), color="C1", alpha=0.4 + 0.15 * k,
             label=f"annealing, {L // 1000}k MD steps (q = {q:.0%} per run)")
    ax2.plot(np.full(sel.sum(), c), E - E_ref, "o", color="C1", alpha=0.35, ms=4,
             label="annealing, one run each" if k == 0 else None)
    ax2.plot(c, np.median(E) - E_ref, "s", color="C1", ms=7, mec="k",
             label="annealing, median" if k == 0 else None)
    print(f"annealing {L} steps: {sel.sum()} runs, {c:.0f} calls each, success {q:.0%},"
          f" best {E.min() - E_ref:+.4f}, median {np.median(E) - E_ref:+.4f}")

# ---- GA: one curve per ecology
hit_calls = []
for r in ga:
    calls, Emin = r["calls"], r["Emin"]
    ax2.plot(calls, Emin - E_ref, color="C2", lw=0.8, alpha=0.7)
    ok = Emin < E_ref + 1e-3
    hit_calls.append(calls[np.argmax(ok)] if ok.any() else np.inf)
hit_calls = np.array(hit_calls)
ax1.step(B, [(hit_calls <= b).mean() for b in B], "C2", where="post", lw=2,
         label=f"genetic algorithm ({np.isfinite(hit_calls).sum()} of {len(ga)} ecologies found C$_{{60}}$)")
ax2.plot([], [], color="C2", label="genetic algorithm, each ecology")
print(f"GA: {len(ga)} ecologies, found in {np.isfinite(hit_calls).sum()};"
      f" calls to find: {np.sort(hit_calls[np.isfinite(hit_calls)]).astype(int).tolist()}")

ax1.set_xscale("log")
ax1.set_xlabel("Energy+force calls")
ax1.set_ylabel("Probability of having found C$_{60}$")
ax1.set_ylim(-0.03, 1.03)
ax1.legend(fontsize=7, loc="upper left")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("Energy+force calls")
ax2.set_ylabel("Best energy above C$_{60}$ (eV/atom)")
ax2.set_ylim(1e-3, 1.5)
ax2.legend(fontsize=7, loc="lower left")
fig.tight_layout()
fig.savefig("step7_compare.png", dpi=150)
print("saved step7_compare.png")
