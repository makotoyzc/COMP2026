"""2D Ising model on an L x L periodic square lattice, for contrast with XY.

    H = -sum_<ij> s_i s_j ,   s_i = +1 or -1

Same lattice, same checkerboard Metropolis, same blocking. The ONLY difference
from xy.py is that the spin is discrete instead of continuous. That one change
is what decides whether the model can order in 2D, so running the two side by
side isolates it.

Exact answers to check against (Onsager 1944), with J = 1, k_B = 1:

    Tc   = 2 / ln(1 + sqrt 2) = 2.269185
    m(T) = (1 - sinh(2/T)^-4)^(1/8)                       for T < Tc, zero above
    E/N  = -coth(2K) [1 + (2/pi)(2 tanh^2(2K) - 1) K(k)],  K = 1/T,
           k = 2 tanh(2K) / cosh(2K)

At Tc the bracket's second term vanishes (tanh(2Kc) = 1/sqrt 2), leaving the
clean number E/N = -sqrt 2.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import ellipk

from xy import checkerboard

TC = 2 / np.log(1 + np.sqrt(2))


def m_exact(T):
    """Onsager spontaneous magnetization; zero at and above Tc."""
    T = np.asarray(T, dtype=float)
    below = T < TC
    out = np.zeros_like(T)
    out[below] = (1 - np.sinh(2 / T[below]) ** -4) ** 0.125
    return out


def e_exact(T):
    """Onsager internal energy per site."""
    K = 1 / np.asarray(T, dtype=float)
    k2 = (2 * np.tanh(2 * K) / np.cosh(2 * K)) ** 2
    # k2 reaches 1 exactly at Tc, where ellipk diverges logarithmically while
    # its prefactor vanishes linearly. The product goes to zero; nudging k2 off
    # 1 evaluates that limit instead of returning 0 * inf.
    term = (2 * np.tanh(2 * K) ** 2 - 1) * ellipk(np.minimum(k2, 1 - 1e-15))
    return -1 / np.tanh(2 * K) * (1 + (2 / np.pi) * term)


def neighbor_sum(s):
    """Sum of the four neighbors' spins at every site."""
    return sum(np.roll(s, sh, ax) for ax in (0, 1) for sh in (1, -1))


def sweep(s, T, masks, rng):
    """One checkerboard sweep. Flipping site i costs dE = 2 s_i h_i."""
    for mask in masks:
        dE = 2 * s * neighbor_sum(s)
        flip = mask & (rng.random(s.shape) < np.exp(-dE / T))
        s = np.where(flip, -s, s)
    return s


def bond_pairs(L):
    """Index pairs of the 2 L^2 nearest-neighbor bonds on a periodic L x L grid."""
    i, j = np.indices((L, L))
    here = (i * L + j).ravel()
    right = (i * L + (j + 1) % L).ravel()
    down = (((i + 1) % L) * L + j).ravel()
    return np.concatenate([here, here]), np.concatenate([right, down])


def sw_sweep(s, T, pairs, rng):
    """One Swendsen-Wang sweep: build clusters from bonds, flip each at random.

    Metropolis flips one spin at a time, so near Tc it needs O(L^2) sweeps to
    move anything -- the L = 128 point would be worthless. Swendsen-Wang
    activates each bond between aligned spins with probability
    p = 1 - exp(-2/T), labels the connected clusters, and flips whole clusters,
    which makes the slowdown almost vanish. `sweep` above is kept because it is
    the same algorithm the XY model uses; `check 0` shows the two agree.
    """
    a, b = pairs
    flat = s.ravel()
    active = (flat[a] == flat[b]) & (rng.random(a.size) < 1 - np.exp(-2 / T))
    n = s.size
    g = coo_matrix((np.ones(active.sum()), (a[active], b[active])), shape=(n, n))
    n_clusters, labels = connected_components(g, directed=False)
    flip = np.where(rng.random(n_clusters) < 0.5, -1, 1)
    return (flat * flip[labels]).reshape(s.shape)


def energy(s):
    """Total energy, counting each bond once."""
    return -(s * np.roll(s, -1, 0)).sum() - (s * np.roll(s, -1, 1)).sum()


def run(L, T, n_therm, n_meas, seed, start="ordered", algo="sw"):
    """Thermalize, then record one measurement per sweep. Picklable worker."""
    rng = np.random.default_rng(seed)
    masks, pairs = checkerboard(L), bond_pairs(L)
    step = (lambda x: sw_sweep(x, T, pairs, rng)) if algo == "sw" else \
           (lambda x: sweep(x, T, masks, rng))
    N = L * L
    s = np.ones((L, L), dtype=int) if start == "ordered" else \
        rng.choice([-1, 1], (L, L))
    for _ in range(n_therm):
        s = step(s)

    rec = {k: np.empty(n_meas) for k in ("E", "m")}
    for n in range(n_meas):
        s = step(s)
        rec["E"][n] = energy(s)
        rec["m"][n] = s.sum() / N
    rec["s"] = s
    return dict(L=L, T=T, rec=rec, N=N)
