"""Figures for step 3, drawn from the saved solutions (no solving here).

step3_bridge.png
    (a) The amplitude half: the self-consistent gap Delta(T), vanishing at the
        mean-field Tc. This is all mean field theory knows.
    (b) The bridge. Each curve is the phase stiffness J(T) that gap implies at
        one sheet resistance. The two straight lines are the two transition
        criteria: the bare Nelson-Kosterlitz T = (pi/2) J, and T = 0.8929 J,
        the coefficient measured in step 2 with vortices included. Where a
        stiffness curve meets a line is that film's transition.
    (c) The answer: T_BKT / Tc_MF against sheet resistance. On the left the
        phase is rigid and the mean-field temperature is the real one. On the
        right the film pairs long before it conducts, and a wide resistive tail
        opens between the two.

Run:  ~/anaconda3/bin/python3 plot_step3.py [results_step3.npz]
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from bcs import stiffness

R_Q = 6.453  # kOhm, h / 4 e^2

data = np.load(sys.argv[1] if len(sys.argv) > 1 else "results_step3.npz")
Ts, Deltas, Tc = data["Ts"], data["Deltas"], data["Tc"]
R, NK, LATTICE = data["R_ratios"], data["NK"], data["LATTICE"]

fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), layout="constrained")

ax = axs[0]
ax.plot(Ts / Tc, Deltas / Deltas[-1])
ax.set_xlabel("T / Tc_MF")
ax.set_ylabel("Delta(T) / Delta(0)")
ax.set_title("(a) Amplitude: the self-consistent gap")

ax = axs[1]
for ratio in (0.2, 1.0, 2.0, 5.0):
    ax.plot(Ts / Tc, stiffness(Deltas, Ts, ratio) / Tc, label=f"R_sq / R_Q = {ratio}")
x = np.linspace(0, 1, 50)
ax.plot(x, x / NK, "k--", label="T = (pi/2) J,  no vortices")
ax.plot(x, x / LATTICE, "k:", label=f"T = {LATTICE} J,  step 2")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.5)
ax.set_xlabel("T / Tc_MF")
ax.set_ylabel("J(T) / Tc_MF")
ax.set_title("(b) Bridge: stiffness meets criterion")
ax.legend(fontsize=8)

ax = axs[2]
ax.semilogx(R, data["T_bkt_NK"] / Tc, "--", label="bare Nelson-Kosterlitz")
ax.semilogx(R, data["T_bkt_lattice"] / Tc, "-", label="with vortices (step 2)")
ax.axhline(1, color="gray", lw=0.8)
ax.set_xlabel("R_sq / R_Q")
ax.set_ylabel("T_BKT / Tc_MF")
ax.set_title("(c) How far coherence falls below pairing")
ax.legend()
secax = ax.secondary_xaxis("top", functions=(lambda r: r * R_Q, lambda s: s / R_Q))
secax.set_xlabel("sheet resistance (kOhm)")

fig.savefig("figures/step3_bridge.png", dpi=150)
print("wrote figures/step3_bridge.png")
