"""Make and load Ising and XY snapshots, using the week-7 samplers.

One chain = one (model, L, T, chain index). It thermalizes from an ordered
start, then saves n_snap snapshots, one every `gap` sweeps. Chains 0-3 are for
training and 4-5 for testing, so train and test never share a chain.

Files: data/<model>_L<L>.npz with
    s      snapshots, shape (nT, n_chain, n_snap, L, L)
           Ising: int8 spins +-1.  XY: float32 angles theta.
    Ts     temperature grid
    absm   |m| per snapshot, shape (nT, n_chain, n_snap)
    e      energy per site per snapshot
    nv     vortex density per snapshot (XY only)
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "07_Stat_Mech_and_Ising" / "XY_BKT"))
import ising  # noqa: E402
import xy  # noqa: E402
from xy_cluster import cluster_step  # noqa: E402

DATA = HERE / "data"
GRIDS = {"ising": np.round(np.arange(1.6, 3.0001, 0.05), 2),
         "xy": np.round(np.arange(0.5, 1.5001, 0.05), 2)}
N_CHAIN, N_SNAP = 6, 150
TRAIN, TEST = [0, 1, 2, 3], [4, 5]
MODEL_ID = {"ising": 0, "xy": 1}


def chain_rng(model, L, iT, c):
    """Fixed, distinct seed for every chain."""
    return np.random.default_rng([MODEL_ID[model], L, iT, c])


def ising_chain(L, iT, c, gap, n_therm):
    T = GRIDS["ising"][iT]
    rng = chain_rng("ising", L, iT, c)
    pairs = ising.bond_pairs(L)
    s = np.ones((L, L), dtype=int)
    for _ in range(n_therm):
        s = ising.sw_sweep(s, T, pairs, rng)
    snaps = np.empty((N_SNAP, L, L), dtype=np.int8)
    for n in range(N_SNAP):
        for _ in range(gap):
            s = ising.sw_sweep(s, T, pairs, rng)
        snaps[n] = s
    absm = np.abs(snaps.reshape(N_SNAP, -1).mean(1))
    e = np.array([ising.energy(x.astype(int)) for x in snaps]) / L**2
    return dict(s=snaps, absm=absm, e=e)


def xy_chain(L, iT, c, gap, n_therm):
    """One "sweep" = one cluster step (xy_cluster.py) + one Metropolis sweep.

    Metropolis alone was too slow (see xy_cluster.py). Its step size delta is
    tuned during thermalization only, as in xy.run.
    """
    T = GRIDS["xy"][iT]
    rng = chain_rng("xy", L, iT, c)
    masks, pairs = xy.checkerboard(L), ising.bond_pairs(L)
    theta = np.zeros((L, L))
    delta = 1.0
    for _ in range(n_therm):
        theta = cluster_step(theta, T, pairs, rng)
        theta, acc = xy.sweep(theta, T, delta, masks, rng)
        delta = min(np.pi, delta * (1.05 if acc > 0.5 else 1 / 1.05))
    snaps = np.empty((N_SNAP, L, L), dtype=np.float32)
    for n in range(N_SNAP):
        for _ in range(gap):
            theta = cluster_step(theta, T, pairs, rng)
            theta, _ = xy.sweep(theta, T, delta, masks, rng)
        snaps[n] = theta
    th = snaps.astype(float)
    absm = np.hypot(np.cos(th).mean((1, 2)), np.sin(th).mean((1, 2)))
    e = np.array([xy.energy(x) for x in th]) / L**2
    nv = np.array([np.abs(xy.vortices(x)).sum() for x in th]) / L**2
    return dict(s=snaps, absm=absm, e=e, nv=nv)


CHAIN = {"ising": ising_chain, "xy": xy_chain}


def _job(args):
    model, L, iT, c, gap, n_therm = args
    return CHAIN[model](L, iT, c, gap, n_therm)


def make(model, L, gap, n_therm, workers=18):
    """Run every (T, chain) for one model and size; save to data/."""
    Ts = GRIDS[model]
    jobs = [(model, L, iT, c, gap, n_therm) for iT in range(len(Ts)) for c in range(N_CHAIN)]
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_job, jobs))
    keys = out[0].keys()
    arrays = {k: np.stack([o[k] for o in out]).reshape(len(Ts), N_CHAIN, *out[0][k].shape)
              for k in keys}
    DATA.mkdir(exist_ok=True)
    np.savez(DATA / f"{model}_L{L}.npz", Ts=Ts, gap=gap, n_therm=n_therm, **arrays)


def load(model, L):
    return dict(np.load(DATA / f"{model}_L{L}.npz"))
