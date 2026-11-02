"""Simulated annealing as molecular dynamics: heat the cluster, cool it slowly, relax.

Langevin dynamics (BAOAB integrator) with the thermostat temperature lowered linearly
from T0 to 0 over the run, then an L-BFGS relaxation of the final geometry. A soft
spherical wall (zero inside R_wall, k (r - R_wall)^2 outside) stops hot atoms from
evaporating; it is off for the final relaxation.

This is the standard annealing method for carbon clusters. The Metropolis moves of
Problem 4 are a weaker version of it: in 180 dimensions a random move of all atoms must be
tiny to be accepted, while MD moves all atoms coherently along the forces.

Units: eV, A, amu; one time unit = sqrt(amu A^2 / eV) = 10.18 fs.
"""
import numpy as np

import carbon_tb as tb
import structures as st

MASS = 12.011            # amu
KB = 8.617333e-5         # eV/K
FS = 1 / 10.1805         # one femtosecond in time units


def wall(R, R_wall, k):
    """Energy and forces of the confining wall around the center of mass."""
    X = R - R.mean(0)
    r = np.linalg.norm(X, axis=1)
    over = np.maximum(r - R_wall, 0.0)
    F = -2 * k * over[:, None] * X / np.maximum(r, 1e-12)[:, None]
    return k * (over**2).sum(), F - F.mean(0)


def anneal(R0, T0, steps, rng, R_wall, dt_fs=1.0, gamma_fs=200.0, k_wall=1.0,
           record_every=0):
    """Langevin MD from R0 with T falling linearly from T0 (K) to 0 over `steps` steps of
    dt_fs; friction time gamma_fs. Returns (R, E_b, calls, info). The final L-BFGS
    relaxation is included in calls."""
    R = np.array(R0, float)
    N = len(R)
    dt, c1 = dt_fs * FS, np.exp(-dt_fs / gamma_fs)
    c2 = np.sqrt(1 - c1**2)
    V = np.zeros_like(R)
    E, F = tb.energy_forces(R)
    Ew, Fw = wall(R, R_wall, k_wall)
    F = F + Fw
    path, Tkin = [], []
    for n in range(steps):
        T = T0 * (1 - n / steps)
        V += 0.5 * dt * F / MASS
        R += 0.5 * dt * V
        V = c1 * V + c2 * np.sqrt(KB * T / MASS) * rng.normal(size=R.shape)
        R += 0.5 * dt * V
        E, F = tb.energy_forces(R)
        Ew, Fw = wall(R, R_wall, k_wall)
        F = F + Fw
        V += 0.5 * dt * F / MASS
        if record_every and n % record_every == 0:
            path.append(R.copy())
            Tkin.append(MASS * (V**2).sum() / (3 * N * KB))
    Rf, Ef, info = st.relax(R, fmax=1e-3, maxiter=4000)
    return Rf, tb.binding(Ef, N), steps + 1 + info["nfev"], dict(path=path, Tkin=Tkin)
