"""Pictures for Step 6 from ga_runs/*.npz.

Usage:  python3 step6_plots.py ga_runs/C60_mu0.0_seed1.npz [more .npz of the same N]
  -> step6_C{N}_energy.png   lowest/highest population energy vs matings (cf. D&H Fig. 1)
     step6_C{N}_folding.png  the best cluster each time it improved (first run given)
     step6_C{N}_folding.gif  the same as a movie
     viewer_step6_C{N}.html  the same in the browser
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

import carbon_tb as tb
import structures as st
import viz

files = sys.argv[1:]
runs = [np.load(f, allow_pickle=True) for f in files]
N = int(runs[0]["N"])
E_t = float(runs[0]["E_target"])
name = "bowl" if N == 20 else "C$_{60}$"

fig, ax = plt.subplots(figsize=(6, 4))
for k, (f, r) in enumerate(zip(files, runs)):
    m = np.arange(1, len(r["Emin"]) + 1)
    lab = (f"seed {int(r['seed'])}, mu {float(r['mu'])}"
           + (", parents not rotated" if "aligned" in f else "")
           + (" (found)" if r["found"] else ""))
    ax.plot(m, r["Emin"], color=f"C{k}", lw=1.2, label=lab)
    ax.plot(m, r["Emax"], color=f"C{k}", lw=0.8, ls=":")
ax.axhline(E_t, color="C3", ls="--", lw=0.8)
ax.text(ax.get_xlim()[1], E_t, name, color="C3", ha="right", va="bottom")
ax.set_xlabel("Mating operations")
ax.set_ylabel("Binding energy (eV/atom)")
ax.set_title(f"C$_{{{N}}}$ GA: population lowest (solid) and highest (dotted)")
lo = min(r["Emin"].min() for r in runs)
ax.set_ylim(lo - 0.05, lo + 0.6)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"step6_C{N}_energy.png", dpi=150)
plt.close(fig)

r = runs[0]
best, best_E, best_at = list(r["best"]), list(r["best_E"]), list(r["best_at"])
pick = np.unique(np.linspace(0, len(best) - 1, min(10, len(best))).round().astype(int))
fig = plt.figure(figsize=(15, 3.3 * int(np.ceil(len(pick) / 5))))
for k, i in enumerate(pick):
    ax = fig.add_subplot(int(np.ceil(len(pick) / 5)), 5, k + 1, projection="3d")
    nr, _ = st.rings(best[i], sizes=(5, 6, 7))
    viz.draw(ax, best[i], f"mating {best_at[i]}: E_b = {best_E[i]:+.4f}\n"
             f"5-rings {nr[5]}, 6-rings {nr[6]}, 7-rings {nr[7]}")
fig.tight_layout()
fig.savefig(f"step6_C{N}_folding.png", dpi=150)
plt.close(fig)

viz.movie(best, best_E, f"step6_C{N}_folding.gif", every=1, fps=4)
viz.viewer([(f"C{N} GA seed {int(r['seed'])}: best so far (step = improvement)", best, best_E)]
           + [(f"C{N} GA seed {int(q['seed'])}: final best", [q["best"][-1]], [q["best_E"][-1]])
              for q in runs[1:]],
           f"viewer_step6_C{N}.html", "improvement")
print("saved", f"step6_C{N}_energy.png, step6_C{N}_folding.png/.gif, viewer_step6_C{N}.html")
