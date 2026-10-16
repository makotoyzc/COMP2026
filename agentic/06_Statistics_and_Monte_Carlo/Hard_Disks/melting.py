"""Step 3: hexagonal order, the order parameter of 2D melting.

THE ORDER PARAMETER
-------------------
In a perfect hexagonal crystal every disk has six nearest neighbors, 60 degrees
apart. For disk j, take the angles theta_jk to its six nearest neighbors and form

    psi6_j = (1/6) sum_k exp(6 i theta_jk)

Multiplying the angle by 6 makes 60-degree steps a full turn, so in a perfect
hexagon all six terms point the same way and |psi6_j| = 1. In a liquid the
neighbor angles are random and the terms mostly cancel. The phase of psi6_j
says which way the local hexagon is rotated (one full turn of the phase is a
60-degree rotation of the hexagon).

The global order parameter averages the complex numbers before taking the size,

    Psi6 = | (1/N) sum_j psi6_j |

so it is near 1 only if the hexagons are good AND all point the same way.
In a liquid it falls to about 1/sqrt(N).

THE STARTING CRYSTAL
--------------------
A hexagonal lattice has rows spaced a*sqrt(3)/2 apart, so it only fits a
periodic box whose sides are nx*a by ny*a*sqrt(3)/2, with ny even so the
staggered rows line up across the edge. Each disk takes area a^2 sqrt(3)/2, so

    eta = (pi/4) / (a^2 sqrt(3)/2)   ->   a = sqrt(pi / (2 sqrt(3) eta))

and a = 1 (disks touching) at the close-packed limit eta = pi/(2 sqrt 3) = 0.907.
"""
import numpy as np

from hard_disks import min_image, sweep, count_overlaps
from pressure import tune_delta, pair_distances


def hex_lattice(nx, ny, eta):
    """Perfect hexagonal crystal at packing fraction eta. Returns (x, L)."""
    assert ny % 2 == 0, "ny must be even for the rows to wrap"
    a = np.sqrt(np.pi / (2 * np.sqrt(3) * eta))
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    x = np.column_stack([(i + 0.5 * (j % 2) + 0.25).ravel() * a,
                         (j + 0.5).ravel() * a * np.sqrt(3) / 2])
    L = np.array([nx * a, ny * a * np.sqrt(3) / 2])
    return x, L


def psi6(x, L):
    """Local hexagonal order psi6_j (complex, one per disk), from six nearest neighbors."""
    d = min_image(x[None, :, :] - x[:, None, :], L)  # d[j, k] points from j to k
    r2 = (d**2).sum(axis=2)
    np.fill_diagonal(r2, np.inf)
    nearest = np.argsort(r2, axis=1)[:, :6]
    dn = np.take_along_axis(d, nearest[:, :, None], axis=1)
    theta = np.arctan2(dn[..., 1], dn[..., 0])
    return np.exp(6j * theta).mean(axis=1)


def squeeze(x, L, eta_target, rng, sweeps_between=2):
    """Compress the box (and every position with it) up to packing fraction eta_target.

    Each round shrinks everything by exactly enough to make the closest pair
    touch, or to reach the target if that comes first. A couple of sweeps then
    open up new room, and the next round shrinks again. No round can create an
    overlap, so nothing is ever undone.
    """
    N = len(x)
    eta = lambda L: N * np.pi / 4 / (L[0] * L[1])
    while eta(L) < eta_target * (1 - 1e-12):
        r_min = pair_distances(x, L).min()
        s = max(np.sqrt(eta(L) / eta_target), (1 + 1e-9) / r_min)
        x, L = x * s, L * s
        for _ in range(sweeps_between):
            sweep(x, L, 0.05, rng)
    assert count_overlaps(x, L) == 0
    return x, L


def run(eta, start, sweeps, seed, nx=16, ny=18, every=20):
    """One independent run at packing fraction eta. Safe to call in a separate process.

    start = "crystal": begin from a perfect hexagonal crystal at eta.
    start = "liquid":  melt a crystal at eta = 0.60, then squeeze it up to eta.
    Returns the Psi6 history (one value every `every` sweeps) and the final state.
    """
    rng = np.random.default_rng(seed)
    if start == "crystal":
        x, L = hex_lattice(nx, ny, eta)
    else:
        x, L = hex_lattice(nx, ny, 0.60)
        for _ in range(2000):
            sweep(x, L, 0.2, rng)
        x, L = squeeze(x, L, eta, rng)

    delta = tune_delta(x, L, rng)
    series = []
    for s in range(sweeps):
        sweep(x, L, delta, rng)
        if s % every == 0:
            series.append(abs(psi6(x, L).mean()))
    assert count_overlaps(x, L) == 0
    return dict(eta=eta, start=start, series=np.array(series), x=x, L=L, delta=delta)
