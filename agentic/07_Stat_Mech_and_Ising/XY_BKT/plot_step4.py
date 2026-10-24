"""Figures for step 4, drawn from the saved runs (no simulation here).

step4_local.png -- what order and disorder look like locally.
    Row 1  XY phases as arrows, on a 24 x 24 window of an L = 64 lattice, with
           vortices circled (+1) and crossed (-1). A vortex is the pinwheel:
           follow the arrows around one and they turn through a full 2 pi.
    Row 2  XY local coherence, the length of the average spin in a 5 x 5 block
           around each site. Bright means locally aligned, dark means scrambled.
    Row 3  Ising spins on the same lattice, for contrast.

    The columns are the same story in each model: well inside the ordered
    phase, near the transition, and above it.

    The thing to see in row 2 is that at T = 0.70 the XY lattice is bright
    nearly everywhere -- locally it is as ordered as the Ising lattice at
    T = 1.8 -- even though its global magnetization is decaying to zero as L
    grows. Local order and no global order, at the same time. That is what
    Mermin-Wagner costs you, and it is why the transition had to be found
    through stiffness instead of through an order parameter.

step4_scaling.png -- a critical point next to a critical line.
    (a) Ising |m|(T) at five sizes against exact Onsager: a curve that
        sharpens toward a step at Tc.
    (b) XY <m^2> against L, log-log, one line per temperature. Straight at
        every T below T_BKT -- a power law everywhere, not just at one point.
    (c) The exponent eta(T) from those slopes, against T / (2 pi Y(T))
        predicted by step 2's stiffness. No fitted parameters. The star is
        eta = 1/4 at T_BKT.
    (d) Specific heat. The Ising peak grows with L because it is a divergence.
        The XY peak does not, and it is not even at T_BKT.

Run:  ~/anaconda3/bin/python3 plot_step4.py [results_step4.npz]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from scaling import local_coherence, local_ising
from xy import vortices

data = np.load(sys.argv[1] if len(sys.argv) > 1 else "results_step4.npz")
TC, T_BKT = float(data["TC"]), float(data["T_BKT"])
XY_LS, XY_TS = list(data["XY_LS"]), list(data["XY_TS"])
IS_LS, IS_TS = list(data["IS_LS"]), list(data["IS_TS"])

# ---- Figure 1: local order ---------------------------------------------------
W = 24  # window of the L = 64 lattice, so single arrows stay readable
xy_cols = [("snap_xy_cold", "T = 0.70, well below T_BKT"),
           ("snap_xy_mid", "T = 0.90, at T_BKT"),
           ("snap_xy_hot", "T = 1.10, above T_BKT")]
is_cols = [("snap_is_cold", "T = 1.80, well below Tc"),
           ("snap_is_mid", "T = 2.269, at Tc"),
           ("snap_is_hot", "T = 2.80, above Tc")]

fig, axs = plt.subplots(3, 3, figsize=(15, 15), layout="constrained")

for col, (key, title) in enumerate(xy_cols):
    theta = data[key]
    v = vortices(theta)[:W, :W]
    th = theta[:W, :W]
    y, x = np.indices(th.shape)
    ax = axs[0, col]
    ax.quiver(x, y, np.cos(th), np.sin(th), th, cmap="twilight",
              clim=(0, 2 * np.pi), pivot="mid", scale=28, width=0.006)
    for q, marker in [(1, "o"), (-1, "x")]:
        iy, ix = np.nonzero(v == q)
        ax.plot(ix + 0.5, iy + 0.5, marker, mfc="none", mec="black", mew=1.8, ms=14)
    ax.set_title(f"XY phases\n{title}")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax = axs[1, col]
    im = ax.imshow(local_coherence(theta, 5), cmap="magma", vmin=0, vmax=1,
                   origin="lower")
    ax.set_title("XY local coherence (5 x 5 blocks)")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axs[1, 2], label="local coherence")

for col, (key, title) in enumerate(is_cols):
    ax = axs[2, col]
    ax.imshow(data[key], cmap="gray", vmin=-1, vmax=1, origin="lower")
    ax.set_title(f"Ising spins\n{title}")
    ax.set_xticks([])
    ax.set_yticks([])

fig.savefig("figures/step4_local.png", dpi=140)
plt.close(fig)

# ---- Figure 2: the scaling contrast ------------------------------------------
from ising import m_exact  # noqa: E402  (only needed for the exact curve)

eta = data["eta"]
fig, axs = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")

ax = axs[0, 0]
for i, L in enumerate(IS_LS):
    ax.errorbar(IS_TS, data["is_m"][i], data["is_dm"][i], fmt="o-", ms=4,
                capsize=2, label=f"L = {L}")
t = np.linspace(min(IS_TS), TC, 200)
ax.plot(t, m_exact(t), "k--", lw=1.5, label="Onsager, exact")
ax.axvline(TC, color="gray", lw=0.8)
ax.set_xlabel("T / J")
ax.set_ylabel("|m|")
ax.set_title("(a) Ising: an order parameter, and a critical point")
ax.legend(fontsize=8)

ax = axs[0, 1]
for j, T in enumerate(XY_TS):
    m2 = np.array(data["xy_m2"])[:, j]
    ax.errorbar(XY_LS, m2, np.array(data["xy_dm2"])[:, j], fmt="o-", ms=4,
                capsize=2, label=f"T = {T:.2f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("L")
ax.set_ylabel("<m^2>")
ax.set_title("(b) XY: a power law at every T, not just one")
ax.legend(fontsize=8)

ax = axs[1, 0]
ax.errorbar(eta[:, 0], eta[:, 1], eta[:, 2], fmt="o", capsize=3,
            label="measured, from <m^2> ~ L^-eta")
ax.plot(eta[:, 0], eta[:, 3], "s--", mfc="none",
        label="predicted, T / (2 pi Y) from step 2")
ax.plot(T_BKT, 0.25, "k*", ms=16, label="eta = 1/4 at T_BKT")
ax.axvline(T_BKT, color="gray", lw=0.8)
ax.set_xlabel("T / J")
ax.set_ylabel("eta")
ax.set_title("(c) The exponent is not free: stiffness predicts it")
ax.legend(fontsize=8)

ax = axs[1, 1]
for i, L in enumerate(IS_LS):
    ax.errorbar(np.array(IS_TS) / TC, data["is_C"][i], data["is_dC"][i],
                fmt="o-", ms=3, capsize=2, label=f"Ising L = {L}")
for i, L in enumerate(data["XY_HEAT_LS"]):
    ax.errorbar(np.array(data["XY_HEAT_TS"]) / T_BKT, data["xy_C"][i],
                data["xy_dC"][i], fmt="s--", ms=4, capsize=2,
                label=f"XY L = {L}")
ax.axvline(1, color="gray", lw=0.8)
ax.set_xlabel("T / T_transition")
ax.set_ylabel("C per site")
ax.set_title("(d) One peak diverges, the other is just a crossover")
ax.legend(fontsize=7, ncol=2)

fig.savefig("figures/step4_scaling.png", dpi=140)
print("wrote figures/step4_local.png and figures/step4_scaling.png")
