"""Hard disks in a periodic box, sampled with Metropolis Monte Carlo.

Units: disk diameter = 1. The box is a square of side L with periodic edges,
so a disk leaving one side comes back in on the other. Every function also
accepts L = np.array([Lx, Ly]) for a rectangular box.

THE RULE
--------
Two disks overlap when their centers are closer than 1. An arrangement with any
overlap has probability 0. Every other arrangement has the same probability.
There is no energy, so temperature never appears.

THE MOVE
--------
Pick a disk at random and propose moving it by a random step, uniform in
[-delta, delta] in x and in y. If the new spot overlaps another disk, reject
the move and leave the disk where it was. Otherwise accept it.

This is Metropolis with A = P(new)/P(old), which here is either 1 or 0, so
"accept with probability min(1, A)" becomes "accept unless it overlaps". The
proposal is symmetric (a step and its reverse are equally likely), and that is
what makes the chain visit every allowed arrangement equally often.
"""
import numpy as np


def box_side(N, eta):
    """Side L of the square box that gives packing fraction eta for N disks."""
    return np.sqrt(N * np.pi / (4 * eta))


def square_grid(N, L):
    """N = n^2 disk centers on an n x n square grid filling the box."""
    n = int(round(np.sqrt(N)))
    assert n * n == N, "N must be a perfect square"
    a = L / n
    assert a > 1, "grid spacing below one diameter: disks would overlap"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    return a * (np.column_stack([i.ravel(), j.ravel()]) + 0.5)


def min_image(d, L):
    """Shortest periodic copy of the separation vector(s) d."""
    return d - L * np.round(d / L)


def sweep(x, L, delta, rng):
    """N attempted single-disk moves, done in place. Returns how many were accepted."""
    N = len(x)
    accepted = 0
    for _ in range(N):
        k = rng.integers(N)
        new = (x[k] + rng.uniform(-delta, delta, 2)) % L
        r2 = (min_image(x - new, L) ** 2).sum(axis=1)
        r2[k] = np.inf  # the disk being moved cannot block itself
        if r2.min() >= 1.0:
            x[k] = new
            accepted += 1
    return accepted


def count_overlaps(x, L):
    """Number of overlapping pairs, checking all pairs at once.

    Written differently from sweep on purpose: it wraps distances with
    min(|d|, L - |d|) instead of rounding, so a bug in one is not silently
    copied into the other. Assumes every center lies in [0, L).
    L is the side of a square box, or (Lx, Ly) for a rectangle.
    """
    Lx, Ly = np.broadcast_to(L, 2)
    dx = np.abs(x[:, None, 0] - x[None, :, 0])
    dy = np.abs(x[:, None, 1] - x[None, :, 1])
    dx = np.minimum(dx, Lx - dx)
    dy = np.minimum(dy, Ly - dy)
    i, j = np.triu_indices(len(x), k=1)
    return int((dx[i, j] ** 2 + dy[i, j] ** 2 < 1.0).sum())
