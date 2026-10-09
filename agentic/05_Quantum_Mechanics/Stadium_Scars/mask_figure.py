"""Draw the mask, so the boundary condition is something you can look at.

Deliberately coarse (h = 1/10) so individual nodes are visible. Three panels:

  (a) which nodes are unknowns.  Filled = mask True = psi solved for. Open = mask
      False = psi is exactly 0. The true stadium wall is drawn on top, and the
      gap between the dots and the curve is the staircase: on the flat walls and
      the symmetry axes the mask lands on the wall exactly, on the arc it cannot.

  (b) where the boundary condition enters the matrix.  Each unknown is coloured by
      how many of its four stencil links were DROPPED because the neighbour is
      not an unknown. 0 = deep interior, the row sums to zero. 1,2,3 = against
      the wall, and the row sums to that many 1/h^2. Those leftovers are the
      whole of "psi = 0 on the wall".

  (c) the zeros in the answer.  The ground state on the full node grid. The grey
      region is not small, not decaying, not approximate -- it is 0.0 in the
      array, because np.zeros put it there and the mask never overwrote it.

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
    """How many of the four neighbours of each node are not unknowns."""
    pad = np.zeros(np.array(mask.shape) + 2, dtype=int)
    pad[1:-1, 1:-1] = mask
    nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
    return np.where(mask, 4 - nb, -1)


def wall(ax):
    """The exact quarter-stadium boundary, as a continuum curve."""
    t = np.linspace(0, np.pi / 2, 200)
    ax.plot([0, 0], [0, R], 'k', lw=1.2)                       # symmetry axis x=0
    ax.plot([0, A], [0, 0], 'k', lw=1.2)                       # symmetry axis y=0
    ax.plot([0, A], [R, R], 'k', lw=1.2)                       # flat wall y=r
    ax.plot(A + R * np.sin(t), R * np.cos(t), 'k', lw=1.2)     # arc
    ax.set_aspect('equal')
    ax.set_xlim(-0.18, A + R + 0.18)
    ax.set_ylim(-0.18, R + 0.18)


mask, X, Y = bq.quarter_stadium_mask(H, A, R)
drop = dropped_links(mask)

fig, axes = plt.subplots(3, 1, figsize=(6.4, 9.6))

ax = axes[0]
ax.plot(X[~mask], Y[~mask], 'o', mfc='none', mec='0.6', ms=5)
ax.plot(X[mask], Y[mask], 'o', color='C0', ms=5)
wall(ax)
ax.set_title(f"(a) the mask at h = 1/10:  {mask.sum()} unknowns, "
             f"{(~mask).size - mask.sum()} nodes held at psi = 0")

ax = axes[1]
counts = {}
for d, c in ((0, '0.8'), (1, 'C0'), (2, 'C1'), (3, 'C3')):
    sel = drop == d
    counts[d] = int(sel.sum())
    ax.plot(X[sel], Y[sel], 'o', color=c, ms=5, label=str(d))
wall(ax)
ax.legend(fontsize=7, loc='upper right', title='links dropped', title_fontsize=7,
          ncol=4, handletextpad=0.1, columnspacing=0.6, framealpha=1.0)
ax.set_title("(b) dropped stencil links = where psi = 0 enters the matrix\n"
             + "   ".join(f"{d}: {counts[d]} nodes" for d in (0, 1, 2, 3))
             + "   (row sum = dropped/h^2)", fontsize=9)

E, psi = bq.solve(mask, H, 1)
g = psi[0] * np.sign(psi[0].sum())          # eigenvector sign is arbitrary
ax = axes[2]
ax.pcolormesh(X, Y, np.where(mask, 0, 1), cmap='Greys', vmin=0, vmax=4,
              shading='nearest')
ax.pcolormesh(X, Y, np.ma.masked_where(~mask, g), cmap='viridis', vmin=0,
              shading='nearest')
wall(ax)
ax.set_title(f"(c) ground state, E = {E[0]:.4f};  grey = exactly 0.0, "
             f"max psi = {g.max():.3f}")

fig.tight_layout()
fig.savefig("figures/mask.png", dpi=150)

print(f"nodes {mask.size}   unknowns {mask.sum()}   zeros {mask.size - mask.sum()}")
print("dropped-link histogram over unknowns:",
      {d: int((drop == d).sum()) for d in (0, 1, 2, 3)})
print(f"psi outside the mask: max |.| = {abs(g[~mask]).max():.1e}  "
      f"(exactly 0.0: {np.all(g[~mask] == 0.0)})")
print("wrote figures/mask.png")
