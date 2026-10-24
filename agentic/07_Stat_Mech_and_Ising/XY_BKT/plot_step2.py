"""Figures for step 2, drawn from the saved runs (no simulation here).

step2_jump.png
    Left: Y(T) at each L against the Nelson-Kosterlitz line 2 T / pi. The
    curves do not drop vertically -- a finite lattice cannot -- but they get
    steeper with L, and where each one cuts the line is T*(L).
    Right: T*(L) against 1 / (ln L)^2. The KT flow is marginal, so the
    finite-size error dies only logarithmically; this is the axis on which that
    error is a straight line, and its intercept is T_BKT.

step2_vortices.png
    L = 32 snapshots below, at, and above T_BKT. theta is an angle, so the
    color map wraps (twilight). Vortices are circled, antivortices crossed.
    Below T_BKT they come only in tight pairs; above it they roam free. The
    fourth panel counts them.

Run:  ~/anaconda3/bin/python3 plot_step2.py [results_step2.npz]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from jump import T_BKT, universal
from xy import vortices

data = np.load(sys.argv[1] if len(sys.argv) > 1 else "results_step2.npz")
Ls, Ts = data["Ls"], data["Ts"]
Tstar, dTstar = data["Tstar"], data["dTstar"]
T0, dT0, c, dc = data["fit"]

# ---- Figure 1: the jump and its extrapolation --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")

ax = axs[0]
for i, L in enumerate(Ls):
    ax.errorbar(Ts, data["Y"][i], data["dY"][i], fmt="o-", capsize=3, ms=4,
                label=f"L = {L}")
t = np.linspace(Ts[0], Ts[-1], 100)
ax.plot(t, universal(t), "k--", label="2 T / pi")
ax.axvline(T_BKT, color="gray", lw=0.8)
ax.plot(Tstar, universal(Tstar), "k*", ms=12, label="T*(L)")
ax.set_xlabel("T / J")
ax.set_ylabel("helicity modulus Y")
ax.set_title("Stiffness crosses the universal line")
ax.legend()

ax = axs[1]
x = 1 / np.log(Ls) ** 2
ax.errorbar(x, Tstar, dTstar, fmt="o", capsize=3, label="T*(L)")
xf = np.linspace(0, x.max() * 1.1, 50)
ax.plot(xf, T0 + c * xf, "-", label=f"T_BKT + c / (ln L)^2,  c = {c:.2f}")
ax.errorbar(0, T0, dT0, fmt="s", capsize=4, label=f"extrapolated {T0:.4f}")
ax.plot(0, T_BKT, "k*", ms=12, label=f"accepted {T_BKT}")
for L, xi, ti in zip(Ls, x, Tstar):
    ax.annotate(f"L = {L}", (xi, ti), textcoords="offset points", xytext=(6, 6))
ax.set_xlabel("1 / (ln L)^2")
ax.set_ylabel("T* / J")
ax.set_title("Logarithmic finite-size correction")
ax.legend()
fig.savefig("figures/step2_jump.png", dpi=150)

# ---- Figure 2: what the vortices are doing -----------------------------------
show = [0.70, 0.90, 1.10]
fig, axs = plt.subplots(1, 4, figsize=(18, 4.4), layout="constrained")
for ax, T in zip(axs, show):
    theta = data[f"theta_32_{T:.2f}"]
    L = theta.shape[0]
    im = ax.imshow(theta, cmap="twilight", vmin=0, vmax=2 * np.pi,
                   origin="lower", extent=(0, L, 0, L))
    v = vortices(theta)
    for q, marker in [(1, "o"), (-1, "x")]:
        i, j = np.nonzero(v == q)
        ax.plot(j + 1, i + 1, marker, mfc="none", mec="white", mew=1.6, ms=9)
    ax.set_title(f"T = {T:.2f}   ({np.abs(v).sum()} vortices)")
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(im, ax=axs[2], label="theta")

ax = axs[3]
for i, L in enumerate(Ls):
    ax.errorbar(Ts, data["n_vort"][i], data["dn_vort"][i], fmt="o-", capsize=3,
                ms=4, label=f"L = {L}")
ax.axvline(T_BKT, color="gray", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel("T / J")
ax.set_ylabel("vortices per site")
ax.set_title("Vortex density")
ax.legend()
fig.savefig("figures/step2_vortices.png", dpi=150)
print("wrote figures/step2_jump.png and figures/step2_vortices.png")
