"""Draw the mask, so the boundary condition is something you can look at.

Coarse on purpose (h = 1/10) so individual grid nodes are visible.

  (a) which nodes are unknowns.  Filled = solved for, open = held at zero. The
      black curve is the real wall. On the three straight sides the open circles
      sit exactly on it. On the arc they cannot, so the mask steps around it.

  (b) how many stencil links each unknown loses.  A node in the middle keeps all
      four neighbours and its matrix row sums to zero. A node against the wall
      keeps three or two, and its row sums to the number it lost, divided by h^2.
      Those leftovers are the only place psi = 0 appears in the calculation.

  (c) the zeros in the answer.  The grey region is not small and not decaying.
      It is 0.0 in the array, because np.zeros put it there and the mask never
      wrote over it.

Run:  python mask_figure.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import billiard as bq

H = 1.0 / 10
A = R = 1.0


def dropped_links(mask):
    """For each unknown, how many of its four neighbours are not unknowns."""
    pad = np.zeros(np.array(mask.shape) + 2, dtype=int)
    pad[1:-1, 1:-1] = mask
    nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    return np.where(mask, 4 - nb, -1)


def wall(ax):
    t = np.linspace(0, np.pi / 2, 200)
    ax.plot([0, 0, A], [R, 0, 0], 'k', lw=1.2)
    ax.plot([0, A], [R, R], 'k', lw=1.2)
    ax.plot(A + R * np.sin(t), R * np.cos(t), 'k', lw=1.2)
    ax.set_aspect('equal')
    ax.set_xlim(-0.16, A + R + 0.16)
    ax.set_ylim(-0.16, R + 0.16)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0]); ax.set_yticks([0, 0.5, 1.0])


mask, X, Y = bq.quarter_stadium_mask(H, A, R)
drop = dropped_links(mask)
counts = {d: int((drop == d).sum()) for d in (0, 1, 2, 3)}

fig, axes = plt.subplots(3, 1, figsize=(6.6, 10.0))

ax = axes[0]
ax.plot(X[~mask], Y[~mask], 'o', mfc='none', mec='0.6', ms=5,
        label=f"held at psi = 0  ({(~mask).sum()} nodes)")
ax.plot(X[mask], Y[mask], 'o', color='C0', ms=5,
        label=f"unknown, solved for  ({mask.sum()} nodes)")
wall(ax)
ax.legend(fontsize=8, loc='lower left', framealpha=1.0)
ax.set_title("(a) The mask decides which nodes are unknowns.\n"
             "There is no third category: everything else is exactly zero.",
             fontsize=10)

ax = axes[1]
for d, c in ((0, '0.8'), (1, 'C0'), (2, 'C1'), (3, 'C3')):
    sel = drop == d
    ax.plot(X[sel], Y[sel], 'o', color=c, ms=5,
            label=f"{d} lost ({counts[d]})")
wall(ax)
ax.legend(fontsize=8, loc='upper right', ncol=4, handletextpad=0.1,
          columnspacing=0.6, framealpha=1.0,
          title="stencil links lost to the wall", title_fontsize=8)
ax.set_title("(b) Each lost link is one term of the stencil going to zero.\n"
             "A row's sum equals the number it lost, over h^2. That is the "
             "boundary condition.", fontsize=10)

E, psi = bq.solve(mask, H, 1)
g = psi[0] * np.sign(psi[0].sum())          # the eigenvector's sign is arbitrary
ax = axes[2]
ax.pcolormesh(X, Y, np.where(mask, 0, 1), cmap='Greys', vmin=0, vmax=4,
              shading='nearest')
im = ax.pcolormesh(X, Y, np.ma.masked_where(~mask, g), cmap='viridis', vmin=0,
                   shading='nearest')
wall(ax)
plt.colorbar(im, ax=ax, label="psi", fraction=0.03, pad=0.02)
ax.set_title(f"(c) Ground state, E = {E[0]:.3f}. Grey is exactly 0.0, not small.\n"
             "Each square is one node, so the colour runs h/2 past the last one.",
             fontsize=10)

fig.tight_layout()
fig.savefig("figures/mask.png", dpi=150)

print(f"grid nodes {mask.size}   unknowns {mask.sum()}   "
      f"held at zero {mask.size - mask.sum()}")
print("links lost to the wall:", counts)
print(f"psi off the table: largest |value| = {abs(g[~mask]).max():.1e}, "
      f"all exactly 0.0: {np.all(g[~mask] == 0.0)}")
print("wrote figures/mask.png")
