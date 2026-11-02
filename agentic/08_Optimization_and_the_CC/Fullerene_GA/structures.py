"""Cluster geometry: building C60, random clusters, local relaxation, bond and ring analysis."""
import itertools

import numpy as np
from scipy.optimize import minimize

import carbon_tb as tb

BOND = 1.8   # A; C-C bonds are 1.2-1.6 A, second neighbors start at 2.3 A


def c60_ideal(edge=1.42):
    """Truncated icosahedron: cyclic permutations of (0, +-1, +-3g), (+-1, +-(2+g), +-2g),
    (+-g, +-2, +-g^3), g the golden ratio. Edge length 2 before scaling."""
    g = (1 + np.sqrt(5)) / 2
    pts = []
    for base in ((0, 1, 3 * g), (1, 2 + g, 2 * g), (g, 2, g**3)):
        for signs in itertools.product((1, -1), repeat=3):
            v = np.array(base) * signs
            for k in range(3):
                pts.append(np.roll(v, k))
    R = np.unique(np.round(pts, 12), axis=0)
    assert len(R) == 60
    return R * edge / 2


def c20_cage(edge=1.45):
    """Dodecahedron: (+-1, +-1, +-1) and cyclic permutations of (0, +-1/g, +-g). Edge 2/g."""
    g = (1 + np.sqrt(5)) / 2
    pts = list(itertools.product((1, -1), repeat=3))
    for a, b in itertools.product((1, -1), repeat=2):
        for k in range(3):
            pts.append(np.roll((0, a / g, b * g), k))
    R = np.array(pts, float)
    assert len(R) == 20
    return R * edge / (2 / g)


def ring(N, bond=1.30):
    """N atoms on a circle, neighbors a distance bond apart."""
    rad = bond / (2 * np.sin(np.pi / N))
    t = 2 * np.pi * np.arange(N) / N
    return rad * np.column_stack([np.cos(t), np.sin(t), np.zeros(N)])


def c20_bowl(c60):
    """Corannulene skeleton cut out of C60: one pentagon and the five hexagons around it."""
    _, found = rings(c60, sizes=(5, 6))
    pent = min(found[5], key=min)
    i, j, _ = bonds(c60)
    edges = [{a, b} for a, b in zip(i, j) if {a, b} <= pent]
    atoms = set(pent)
    for h in found[6]:
        if any(e <= h for e in edges):
            atoms |= h
    assert len(atoms) == 20
    return c60[sorted(atoms)]


def random_cluster(N, rng, radius, dmin=1.2):
    """N points uniform in a sphere of the given radius, no two closer than dmin."""
    R = []
    while len(R) < N:
        p = rng.uniform(-radius, radius, 3)
        if p @ p <= radius**2 and all(np.linalg.norm(p - q) >= dmin for q in R):
            R.append(p)
    return np.array(R)


def relax(R, fmax, maxiter=5000, record=False):
    """L-BFGS down to the nearest local minimum, stopping when every force component is
    below fmax (eV/A). Returns (R, E, info); info['path'] holds every iterate if record."""
    path = [np.array(R, float)]

    def fun(x):
        E, F = tb.energy_forces(x.reshape(-1, 3))
        return E, -F.ravel()

    res = minimize(fun, np.ravel(R), jac=True, method="L-BFGS-B",
                   callback=(lambda x: path.append(x.reshape(-1, 3).copy())) if record else None,
                   options=dict(gtol=fmax, ftol=0.0, maxiter=maxiter, maxcor=30))
    Rf = res.x.reshape(-1, 3)
    info = dict(converged=res.success, message=res.message, nit=res.nit, nfev=res.nfev,
                fmax=np.abs(res.jac).max(), path=path if record else None)
    return Rf, res.fun, info


def bonds(R, cut=BOND):
    """Bonded pairs i < j with r_ij < cut, and their lengths."""
    i, j = np.triu_indices(len(R), 1)
    r = np.linalg.norm(R[j] - R[i], axis=1)
    keep = r < cut
    return i[keep], j[keep], r[keep]


def rings(R, sizes=(3, 4, 5, 6, 7), cut=BOND):
    """Count the simple cycles of each length in the bond graph (each cycle once).
    For a fullerene the 5- and 6-cycles are exactly the pentagonal and hexagonal faces."""
    i, j, _ = bonds(R, cut)
    nbr = [set() for _ in R]
    for a, b in zip(i, j):
        nbr[a].add(b)
        nbr[b].add(a)
    found = {n: set() for n in sizes}
    nmax = max(sizes)

    def walk(path):
        for b in nbr[path[-1]]:
            if b == path[0] and len(path) in found:
                found[len(path)].add(frozenset(path))
            elif b > path[0] and b not in path and len(path) < nmax:
                walk(path + [b])

    for a in range(len(R)):
        walk([a])            # start at the lowest-index atom of each cycle
    return {n: len(c) for n, c in found.items()}, found


def coordination(R, cut=BOND):
    i, j, _ = bonds(R, cut)
    return np.bincount(i, minlength=len(R)) + np.bincount(j, minlength=len(R))


def components(R, cut=BOND):
    """Number of connected pieces of the bond graph."""
    i, j, _ = bonds(R, cut)
    label = np.arange(len(R))
    for _ in range(len(R)):                      # propagate the smallest label along bonds
        new = label.copy()
        np.minimum.at(new, i, label[j])
        np.minimum.at(new, j, label[i])
        if (new == label).all():
            break
        label = new
    return len(np.unique(label))


def classify(R):
    """Structure class from the bond graph:
    fragments  more than one piece
    chain      no rings at all
    ring       every atom has 2 bonds (one big monocyclic ring)
    cage       every atom has 3 bonds (closed, fullerene-like)
    sp3        some atom has 4 or more bonds
    flat / bowl  everything else (rings with edges), split by how far from a plane"""
    if components(R) > 1:
        return "fragments"
    z = coordination(R)
    if (z == 2).all():
        return "ring"
    if (z == 3).all():
        return "cage"
    if z.max() >= 4:
        return "sp3"
    if len(bonds(R)[0]) - len(R) + 1 == 0:      # independent cycles = bonds - atoms + pieces
        return "chain"
    X = R - R.mean(0)
    out_of_plane = np.sqrt(np.linalg.eigvalsh(X.T @ X / len(R))[0])
    return "flat" if out_of_plane < 0.25 else "bowl"


def describe(R):
    """One-line summary: binding energy, gap, coordination histogram, ring counts."""
    E = tb.energy(R)
    _, gap = tb.levels(R)
    z = np.bincount(coordination(R), minlength=5)
    nr, _ = rings(R)
    zs = " ".join(f"z{k}:{z[k]}" for k in range(len(z)) if z[k])
    rs = " ".join(f"{n}-ring:{c}" for n, c in nr.items() if c)
    return f"E_b = {tb.binding(E, len(R)):+.4f} eV/atom, gap = {gap:.3f} eV, {zs}, {rs}"
