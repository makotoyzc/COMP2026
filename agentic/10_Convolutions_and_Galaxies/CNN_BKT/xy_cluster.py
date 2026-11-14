"""Cluster update for the XY model: Wolff's reflection trick, done Swendsen-Wang style.

Local Metropolis is too slow for this project: at L = 32, snapshots 50 sweeps
apart still had a |m| correlation of 0.5-0.8 (Step 0, first try).

The move (Wolff 1989):
    pick a random direction r = (cos phi, sin phi)
    project every spin on it:  p_i = cos(theta_i - phi)
    the signs of p_i are an Ising model with couplings |p_i p_j| / T
    build Swendsen-Wang clusters of that Ising model: a bond with p_i p_j > 0
        is active with probability 1 - exp(-2 p_i p_j / T)
    reflect each cluster with probability 1/2:  theta -> pi + 2 phi - theta
        (this flips p_i and keeps the component perpendicular to r)

It reuses ising.bond_pairs, the same bond list as ising.sw_sweep.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def cluster_step(theta, T, pairs, rng):
    a, b = pairs
    phi = rng.uniform(0, 2 * np.pi)
    p = np.cos(theta.ravel() - phi)
    pp = p[a] * p[b]
    active = (pp > 0) & (rng.random(a.size) < 1 - np.exp(-2 * pp / T))
    n = theta.size
    g = coo_matrix((np.ones(active.sum()), (a[active], b[active])), shape=(n, n))
    n_clusters, labels = connected_components(g, directed=False)
    flip = (rng.random(n_clusters) < 0.5)[labels]
    flat = np.where(flip, np.pi + 2 * phi - theta.ravel(), theta.ravel())
    return (flat % (2 * np.pi)).reshape(theta.shape)
