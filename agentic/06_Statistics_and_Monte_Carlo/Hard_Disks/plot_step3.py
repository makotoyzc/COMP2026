"""Figures for step 3, drawn from the saved runs (no simulation here).

Heat-map colors are centered on the main crystal: each disk's hexagon
orientation is measured relative to the box-wide average orientation (the phase
of the global Psi6). So the main crystal is always the center color, and any
grain rotated away from it stands out. Orientation repeats every 60 degrees,
so the scale runs from -30 to +30 and wraps around.

Run:  ~/anaconda3/bin/python3 plot_step3.py [results_step3.npz]
Writes figures/step3_order.png and figures/step3_heatmaps.png.
"""
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

from melting import psi6

EVERY = 20  # sweeps between saved Psi6 values, as in melting.run
data = np.load(sys.argv[1] if len(sys.argv) > 1 else "results_step3.npz")
etas = list(data["etas"])
get = lambda start, eta, what: data[f"{start}_{eta:.2f}_{what}"]
second_half = lambda s: s[len(s) // 2:]
N = len(get("crystal", etas[0], "x"))

# ---- Figure 1: order parameter vs eta, and its history ------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
ax = axs[0]
for start, fmt in [("crystal", "o-"), ("liquid", "s--")]:
    tails = [second_half(get(start, e, "series")) for e in etas]
    ax.errorbar(etas, [t.mean() for t in tails], [t.std() for t in tails],
                fmt=fmt, capsize=3, label=f"from {start}")
ax.axhline(1 / np.sqrt(N), color="gray", linestyle=":", label="1/sqrt(N), no order")
ax.set_xlabel("packing fraction eta")
ax.set_ylabel("Psi6 (2nd half of run)")
ax.set_title(f"Hexagonal order parameter, N = {N}")
ax.legend()

ax = axs[1]
for eta, color in [(0.68, "C0"), (0.70, "C1"), (0.72, "C2")]:
    for start, style in [("crystal", "-"), ("liquid", "--")]:
        s = get(start, eta, "series")
        ax.plot(np.arange(len(s)) * EVERY, s, style, color=color, lw=0.8,
                label=f"eta = {eta:.2f}, from {start}")
ax.set_xlabel("sweep")
ax.set_ylabel("Psi6")
ax.set_title("History: does each run forget its start?")
ax.legend(fontsize=8)
fig.savefig("figures/step3_order.png", dpi=150)


# ---- Figure 2: heat maps ---------------------------------------------------------------
def draw(ax, x, L, values, cmap, vmin, vmax):
    shifts = [(sx, sy) for sx in (-L[0], 0, L[0]) for sy in (-L[1], 0, L[1])]
    circles = [Circle((cx + sx, cy + sy), 0.5) for cx, cy in x for sx, sy in shifts]
    pc = PatchCollection(circles, cmap=cmap, edgecolor="none")
    pc.set_array(np.tile(values[:, None], (1, len(shifts))).ravel())
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(0, L[0])
    ax.set_ylim(0, L[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return pc


show = [0.68, 0.70, 0.72, 0.74]
rows = [("crystal", "angle"), ("liquid", "angle"), ("liquid", "size")]
fig, axs = plt.subplots(len(rows), len(show), figsize=(3.6 * len(show) + 1.5, 3.6 * len(rows)),
                        layout="constrained")
for col, eta in enumerate(show):
    for row, (start, what) in enumerate(rows):
        x, L = get(start, eta, "x"), get(start, eta, "L")
        p = psi6(x, L)
        if what == "angle":
            relative = np.degrees(np.angle(p * np.conj(p.mean()))) / 6
            pc_angle = draw(axs[row, col], x, L, relative, "twilight_shifted", -30, 30)
        else:
            pc_size = draw(axs[row, col], x, L, np.abs(p), "viridis", 0, 1)
        if col == 0:
            label = f"from {start}\n" + ("orientation" if what == "angle" else "local order")
            axs[row, col].set_ylabel(label)
    axs[0, col].set_title(f"eta = {eta:.2f}")
fig.colorbar(pc_angle, ax=axs[:2], label="hexagon orientation\nrelative to the main crystal (degrees)",
             shrink=0.6)
fig.colorbar(pc_size, ax=axs[2], label="local order |psi6|", shrink=0.9)
fig.savefig("figures/step3_heatmaps.png", dpi=150)
print("wrote figures/step3_order.png and figures/step3_heatmaps.png")
