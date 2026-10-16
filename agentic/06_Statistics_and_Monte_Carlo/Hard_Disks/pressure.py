"""Step 2: the pressure of hard disks, read off from how often they touch.

WHY TOUCHING GIVES PRESSURE
---------------------------
Hard disks only push on each other at the instant they touch, so the pressure
is set by how many pairs sit right at contact (distance 1). That count comes
from the pair distribution g(r):

    g(r) = (pairs found at distance r) / (pairs an ideal gas would have there)

so g = 1 means "no correlation", and g = 0 inside r < 1 because disks can't
overlap. In 2D the pressure follows from the value of g just outside contact:

    Z = P A / (N k T) = 1 + 2 eta g(1+)

Z = 1 is the ideal gas. Everything above 1 comes from disks bumping.

HOW g(1+) IS MEASURED
---------------------
Histogram every pair distance in bins of width dr, divide by the ideal-gas
count, then fit a quadratic to the bins with 1 < r < 1.2 and evaluate it at
r = 1. The fit is needed because the first bin's center sits at 1 + dr/2, not
at 1, and g falls steeply there at high density.

ERROR BARS
----------
Consecutive sweeps are strongly correlated, so sigma/sqrt(n_sweeps) would be
far too small. Instead the run is cut into blocks of many sweeps, Z is measured
separately in each block, and the error bar is the spread of the block values
divided by sqrt(n_blocks). This is the notebook's Problem 4, with blocks in
place of independent samples.
"""
import numpy as np

from hard_disks import box_side, square_grid, min_image, sweep

FIT_MAX = 1.2  # contact extrapolation uses bins with 1 < r < FIT_MAX


def pair_distances(x, L):
    """All N(N-1)/2 periodic pair distances."""
    i, j = np.triu_indices(len(x), k=1)
    return np.hypot(*min_image(x[i] - x[j], L).T)


def tune_delta(x, L, rng, sweeps=200, target=0.4):
    """Adjust the step size toward `target` acceptance. Warm-up only.

    Changing delta while measuring would break the symmetry of the proposal,
    so delta is frozen once this returns.
    """
    delta = 0.5
    for _ in range(sweeps):
        rate = sweep(x, L, delta, rng) / len(x)
        delta = min(delta * np.exp(rate - target), np.min(L) / 2)
    return delta


def g_of_r(counts, edges, N, L, n_configs):
    """Pair counts in each shell divided by the ideal-gas expectation."""
    shell_area = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    ideal = n_configs * N * (N - 1) / 2 * shell_area / L**2
    return counts / ideal


def contact_value(g, edges):
    """g(1+): quadratic fit to the bins with 1 < r < FIT_MAX, evaluated at r = 1."""
    centers = 0.5 * (edges[1:] + edges[:-1])
    near = (centers > 1) & (centers < FIT_MAX)
    return np.polyval(np.polyfit(centers[near], g[near], 2), 1.0)


def run(eta, rng, N=256, warmup=500, n_blocks=20, block_sweeps=400, dr=0.02):
    """Equilibrate at packing fraction eta, then measure g(r) and Z in blocks."""
    L = box_side(N, eta)
    x = square_grid(N, L)
    delta = tune_delta(x, L, rng)
    for _ in range(warmup):
        sweep(x, L, delta, rng)

    edges = dr * np.arange(int(L / 2 / dr) + 1)  # 0 to L/2: the r < 1 bins are a check
    block_counts = np.zeros((n_blocks, len(edges) - 1))
    accepted = 0
    for b in range(n_blocks):
        for _ in range(block_sweeps):
            accepted += sweep(x, L, delta, rng)
            block_counts[b] += np.histogram(pair_distances(x, L), edges)[0]

    Z_blocks = np.array([1 + 2 * eta * contact_value(g_of_r(c, edges, N, L, block_sweeps), edges)
                         for c in block_counts])
    g = g_of_r(block_counts.sum(axis=0), edges, N, L, n_blocks * block_sweeps)
    return dict(eta=eta, L=L, delta=delta, acceptance=accepted / (n_blocks * block_sweeps * N),
                edges=edges, g=g, counts=block_counts.sum(axis=0), Z_blocks=Z_blocks,
                Z=Z_blocks.mean(), dZ=Z_blocks.std(ddof=1) / np.sqrt(n_blocks))
