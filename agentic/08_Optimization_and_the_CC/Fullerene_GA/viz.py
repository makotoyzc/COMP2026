"""Pictures of clusters: 3D ball-and-stick plots, relaxation movies, and a browser viewer.

Colors: atoms by number of bonds (2 = chain, 3 = sp2 like graphite and C60, 4 = sp3 like
diamond); bonds in red when they lie on a pentagon, gray otherwise.
"""
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import carbon_tb as tb
import structures as st

ZCOLOR = {0: "0.6", 1: "0.6", 2: "C2", 3: "C0", 4: "C1"}   # green chain, blue sp2, orange sp3
ZCOLOR_HEX = {0: "#999999", 1: "#999999", 2: "#2ca02c", 3: "#1f77b4", 4: "#ff7f0e"}


def pentagon_bonds(R):
    i, j, _ = st.bonds(R)
    _, found = st.rings(R, sizes=(5,))
    return i, j, np.array([any({a, b} <= ring for ring in found[5]) for a, b in zip(i, j)],
                          dtype=bool)


def draw(ax, R, title="", lim=None):
    """Ball-and-stick plot of R on a 3D axis."""
    R = R - R.mean(0)
    i, j, pent = pentagon_bonds(R)
    for a, b, p in zip(i, j, pent):
        ax.plot(*R[[a, b]].T, color="C3" if p else "0.55", lw=1.6 if p else 1.0)
    z = st.coordination(R)
    ax.scatter(*R.T, s=22, c=[ZCOLOR[min(k, 4)] for k in z], depthshade=True,
               edgecolors="k", linewidths=0.3)
    lim = lim or 1.05 * np.abs(R).max()
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def snapshots(runs, fname, E_ref):
    """runs: list of (label, path, energies). One row per run: start, middle, end, E vs step.
    The energy axis is zoomed to [E_ref - 0.2, E_ref + 3] eV/atom; E_ref is dashed."""
    fig = plt.figure(figsize=(12, 3.3 * len(runs)))
    for row, (label, path, Es) in enumerate(runs):
        lim = 1.05 * max(np.abs(p - p.mean(0)).max() for p in path)
        for col, k in enumerate((0, len(path) // 4, len(path) - 1)):
            ax = fig.add_subplot(len(runs), 4, 4 * row + col + 1, projection="3d")
            draw(ax, path[k], f"{label}, step {k}\n" + short(path[k]), lim)
        ax = fig.add_subplot(len(runs), 4, 4 * row + 4)
        ax.plot(Es, "k.-", ms=3, lw=0.8)
        ax.axhline(E_ref, color="C3", ls="--", lw=0.8)
        ax.text(len(Es) - 1, E_ref + 0.05, "C$_{60}$", color="C3", ha="right", va="bottom")
        ax.set_ylim(E_ref - 0.2, E_ref + 3)
        ax.set_xlabel("L-BFGS step")
        ax.set_ylabel("Binding energy (eV/atom)")
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"saved {fname}")


def short(R):
    """Energy and ring counts in two short lines for a plot title."""
    nr, _ = st.rings(R, sizes=(5, 6))
    z = np.bincount(st.coordination(R), minlength=5)
    return (f"E_b = {tb.binding(tb.energy(R), len(R)):+.3f} eV/atom\n"
            f"5-rings {nr[5]}, 6-rings {nr[6]}, sp2 atoms {z[3]}")


def movie(path, Es, fname, every=1, fps=12):
    """Rotating GIF of a relaxation path."""
    frames = list(range(0, len(path), every)) + [len(path) - 1] * fps   # hold the last frame
    lim = 1.05 * max(np.abs(p - p.mean(0)).max() for p in path)
    fig = plt.figure(figsize=(4.5, 4.8))
    ax = fig.add_subplot(projection="3d")

    def update(n):
        ax.cla()
        k = frames[n]
        draw(ax, path[k], f"step {k}:  E_b = {Es[k]:+.3f} eV/atom", lim)
        ax.view_init(elev=20, azim=30 + 2 * n)

    FuncAnimation(fig, update, frames=len(frames)).save(fname, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved {fname}")


def viewer(runs, fname, unit, template="viewer_template.html", max_frames=150):
    """Write a self-contained HTML viewer (drag to rotate, slider for the step).
    runs: list of (label, path, energies); unit names one frame (e.g. "L-BFGS step");
    long paths are thinned to about max_frames, always keeping the last frame."""
    data = []
    for label, path, Es in runs:
        keep = sorted(set(range(0, len(path), max(1, len(path) // max_frames))) | {len(path) - 1})
        frames = []
        for k in keep:
            R, E = path[k], Es[k]
            R = R - R.mean(0)
            i, j, pent = pentagon_bonds(R)
            z = st.coordination(R)
            frames.append(dict(x=np.round(R, 3).tolist(), E=round(float(E), 4), n=k,
                               b=[[int(a), int(b), int(p)] for a, b, p in zip(i, j, pent)],
                               c=[ZCOLOR_HEX[min(k, 4)] for k in z]))
        data.append(dict(label=label, frames=frames, unit=unit))
    html = open(template).read().replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    open(fname, "w").write(html)
    print(f"saved {fname}")
