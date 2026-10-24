"""Step 4: finite-size scaling, and what local order looks like in each model.

THE CONTRAST
------------
Ising and XY live on the same lattice and are sampled by the same algorithm.
The only difference is discrete versus continuous spin. That difference decides:

  Ising   a critical POINT. Below Tc the magnetization is finite and stays
          finite as L grows. At Tc it decays as L^(-beta/nu) = L^(-1/8).
          Everywhere else the decay is not a power law at all.

  XY      a critical LINE. There is no long-range order at any T > 0, so
          <m^2> decays as a power of L at EVERY temperature below T_BKT,
          with a temperature-dependent exponent:

              <m^2> ~ L^(-eta(T)),   eta(T) = T / (2 pi Y(T))

          and eta = 1/4 exactly at T_BKT, since Y(T_BKT) = 2 T_BKT / pi there.

The 1/4 is the point of this step. It is universal, it is exact, and it comes
from correlations rather than from the stiffness -- so it tests step 2's answer
with an observable step 2 never touched. The Y(T) measured in step 2 predicts
eta(T) at every temperature with no free parameters.

LOCAL ORDER
-----------
Global order is one number for the whole lattice and hides what is actually
happening. The local coherence

    C_w(i) = | average of exp(i theta) over a w x w block centered on i |

is 1 where the phase is locally aligned and 0 where it is scrambled. In the XY
low-temperature phase it is near 1 EVERYWHERE while the global magnetization
still decays to zero with L: the system is ordered locally and disordered
globally, at the same time. That combination is impossible in the Ising model
and is the clearest single picture of what Mermin-Wagner actually means.
"""
import numpy as np

from xy import run as xy_run, blocks, stiffness, mean_err
import ising


def local_coherence(theta, w=5):
    """Magnitude of the average spin in a w x w neighborhood of every site."""
    z = np.exp(1j * theta)
    s = sum(np.roll(np.roll(z, di, 0), dj, 1)
            for di in range(-(w // 2), w // 2 + 1)
            for dj in range(-(w // 2), w // 2 + 1))
    return np.abs(s) / w ** 2


def local_ising(s, w=5):
    """The same quantity for Ising spins, so the two pictures are comparable."""
    t = sum(np.roll(np.roll(s, di, 0), dj, 1)
            for di in range(-(w // 2), w // 2 + 1)
            for dj in range(-(w // 2), w // 2 + 1))
    return np.abs(t) / w ** 2


def heat(E_block, T, N):
    """Specific heat per site from one block of energies."""
    return E_block.var() / (N * T ** 2)


def xy_measure(L, T, n_therm, n_meas, seed, start="ordered", n_blocks=20):
    """One XY run reporting <m^2>, specific heat and stiffness. Picklable."""
    rng = np.random.default_rng(seed)
    N = L * L
    rec = xy_run(L, T, n_therm, n_meas, rng, start)
    bl = blocks(rec, n_blocks)
    m2, dm2 = mean_err([((b["mx"] ** 2 + b["my"] ** 2) / N ** 2).mean() for b in bl])
    C, dC = mean_err([heat(b["E"], T, N) for b in bl])
    Y, dY = mean_err([stiffness(b, T, N) for b in bl])
    E, dE = mean_err([b["E"].mean() / N for b in bl])
    return dict(model="xy", L=L, T=T, start=start, m2=m2, dm2=dm2, C=C, dC=dC,
                Y=Y, dY=dY, E=E, dE=dE, config=rec["theta"])


def ising_measure(L, T, n_therm, n_meas, seed, n_blocks=20):
    """One Ising run reporting |m|, specific heat and energy. Picklable."""
    r = ising.run(L, T, n_therm, n_meas, seed)
    N, rec = r["N"], r["rec"]
    n = n_meas // n_blocks
    bl = [{k: rec[k][b * n:(b + 1) * n] for k in ("E", "m")} for b in range(n_blocks)]
    m, dm = mean_err([np.abs(b["m"]).mean() for b in bl])
    m2, dm2 = mean_err([(b["m"] ** 2).mean() for b in bl])
    C, dC = mean_err([heat(b["E"], T, N) for b in bl])
    E, dE = mean_err([b["E"].mean() / N for b in bl])
    return dict(model="ising", L=L, T=T, m=m, dm=dm, m2=m2, dm2=dm2,
                C=C, dC=dC, E=E, dE=dE, config=r["rec"]["s"])


def power_fit(Ls, y, dy):
    """Slope of log y against log L, with its error. Returns (exponent, error).

    y ~ L^(-exponent), so the exponent is minus the slope.
    """
    Ls, y, dy = np.asarray(Ls, float), np.asarray(y), np.asarray(dy)
    (slope, _), cov = np.polyfit(np.log(Ls), np.log(y), 1, w=y / dy, cov=True)
    return -slope, np.sqrt(cov[0, 0])
