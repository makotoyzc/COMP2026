"""2D XY model on an L x L periodic square lattice, sampled with Metropolis.

Units: J = 1, k_B = 1. Each site holds an angle theta in [0, 2 pi).

    H = -sum_<ij> cos(theta_i - theta_j)      (nearest neighbors)

THE MOVE
--------
Checkerboard Metropolis. Color the lattice like a chessboard; a site's four
neighbors all have the other color, so every site of one color can be updated
at once. Each site proposes theta + a uniform step in [-delta, delta] and
accepts with probability min(1, exp(-dE / T)). L must be even so the coloring
survives the periodic edges.

The energy of site i in the field of its neighbors is
-(hx cos theta_i + hy sin theta_i), where (hx, hy) is the sum of the
neighbors' unit vectors.
"""
import numpy as np

KEYS = ("E", "cx", "sx", "cy", "sy", "n_vort", "q", "mx", "my")


def checkerboard(L):
    """The two color masks of an L x L lattice."""
    assert L % 2 == 0, "L must be even for the checkerboard update"
    i, j = np.indices((L, L))
    even = (i + j) % 2 == 0
    return even, ~even


def neighbor_field(theta):
    """Sum of the four neighbors' unit vectors at every site."""
    hx = np.zeros_like(theta)
    hy = np.zeros_like(theta)
    for axis in (0, 1):
        for shift in (1, -1):
            nb = np.roll(theta, shift, axis)
            hx += np.cos(nb)
            hy += np.sin(nb)
    return hx, hy


def sweep(theta, T, delta, masks, rng):
    """One checkerboard sweep. Returns (new theta, acceptance fraction)."""
    accepted = 0
    for mask in masks:
        hx, hy = neighbor_field(theta)
        new = theta + rng.uniform(-delta, delta, theta.shape)
        dE = -(hx * np.cos(new) + hy * np.sin(new)) + (hx * np.cos(theta) + hy * np.sin(theta))
        accept = mask & (rng.random(theta.shape) < np.exp(-dE / T))
        theta = np.where(accept, new, theta)
        accepted += accept.sum()
    return theta % (2 * np.pi), accepted / theta.size


def bond_sums(theta):
    """For x and y bonds: sum of cos(d) and sum of sin(d), d = theta_i - theta_(i+1)."""
    out = []
    for axis in (1, 0):  # axis 1 = x (columns), axis 0 = y (rows)
        d = theta - np.roll(theta, -1, axis)
        out.append((np.cos(d).sum(), np.sin(d).sum()))
    return out


def energy(theta):
    """Total energy."""
    (cx, _), (cy, _) = bond_sums(theta)
    return -(cx + cy)


def wrap(a):
    """Angle wrapped to [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def vortices(theta):
    """Winding number (-1, 0 or +1) of each plaquette.

    Plaquette (i, j) has corners (i, j) -> (i, j+1) -> (i+1, j+1) -> (i+1, j).
    """
    a = theta
    b = np.roll(theta, -1, 1)
    c = np.roll(b, -1, 0)
    d = np.roll(theta, -1, 0)
    total = wrap(b - a) + wrap(c - b) + wrap(d - c) + wrap(a - d)
    return np.rint(total / (2 * np.pi)).astype(int)


def run(L, T, n_therm, n_meas, rng, start="ordered"):
    """Thermalize, then record one measurement per sweep.

    delta is tuned during thermalization only, toward 50% acceptance, so the
    measured chain uses a fixed proposal.

    Returns a dict of per-sweep arrays: E (total energy), cx, sx, cy, sy
    (bond sums), n_vort (number of vortices plus antivortices), q (net
    vorticity), mx and my (the two components of the total spin), and the
    final theta and delta.
    """
    masks = checkerboard(L)
    if start == "ordered":
        theta = np.zeros((L, L))
    else:
        theta = rng.uniform(0, 2 * np.pi, (L, L))
    delta = 1.0
    for _ in range(n_therm):
        theta, acc = sweep(theta, T, delta, masks, rng)
        delta = min(np.pi, delta * (1.05 if acc > 0.5 else 1 / 1.05))

    rec = {k: np.empty(n_meas) for k in KEYS}
    for n in range(n_meas):
        theta, _ = sweep(theta, T, delta, masks, rng)
        (cx, sx), (cy, sy) = bond_sums(theta)
        v = vortices(theta)
        rec["E"][n] = -(cx + cy)
        rec["cx"][n], rec["sx"][n], rec["cy"][n], rec["sy"][n] = cx, sx, cy, sy
        rec["n_vort"][n] = np.abs(v).sum()
        rec["q"][n] = v.sum()
        rec["mx"][n], rec["my"][n] = np.cos(theta).sum(), np.sin(theta).sum()
    rec["theta"] = theta
    rec["delta"] = delta
    return rec


def stiffness(rec, T, N):
    """Helicity modulus per site from one block of measurements, x and y averaged."""
    Yx = (rec["cx"].mean() - (rec["sx"] ** 2).mean() / T) / N
    Yy = (rec["cy"].mean() - (rec["sy"] ** 2).mean() / T) / N
    return 0.5 * (Yx + Yy)


def blocks(rec, n_blocks=20):
    """Split per-sweep records into n_blocks consecutive blocks."""
    n = len(rec["E"]) // n_blocks
    return [{k: rec[k][b * n:(b + 1) * n] for k in KEYS} for b in range(n_blocks)]


def mean_err(values):
    """Mean and standard error of block values."""
    values = np.asarray(values)
    return values.mean(), values.std(ddof=1) / np.sqrt(len(values))
