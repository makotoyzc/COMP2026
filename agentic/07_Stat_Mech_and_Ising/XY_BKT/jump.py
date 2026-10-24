"""Step 2: the universal jump of the helicity modulus.

THE PREDICTION
--------------
In 2D the XY model has no true long-range order at any T > 0, so the usual
order parameter is useless. What does order the low-T phase is stiffness: the
free-energy cost of twisting the phase across the system. That is the helicity
modulus Y. Kosterlitz and Thouless, and then Nelson and Kosterlitz (1977),
showed that Y does not go to zero continuously. It drops from a universal value
straight to zero:

    Y(T_BKT^-) = 2 T_BKT / pi ,      Y(T_BKT^+) = 0 .

The argument in one line: a vortex pair at separation r costs energy
2 pi Y ln r and gains entropy 4 T ln r (the pair's center can sit in r^2 places,
roughly), so pairs unbind exactly when 2 pi Y = 4 T. Nothing about J or the
lattice survives, which is why the number is universal.

FINDING IT ON A FINITE LATTICE
------------------------------
A jump needs an infinite system. On L x L the drop is smooth, and the standard
estimator is where the measured Y(T) crosses the line 2 T / pi:

    Y(T*) = 2 T* / pi .

T*(L) sits above T_BKT and comes down as L grows, but only like 1 / (ln L)^2 --
the logarithmic correction from the marginal flow of the KT equations. So we
measure T*(L) at several L and fit

    T*(L) = T_BKT + c / (ln L)^2 .

The intercept of that straight line (against 1 / (ln L)^2) is the estimate of
T_BKT, to be compared with the accepted 0.8929.
"""
import numpy as np

from xy import run, blocks, stiffness, mean_err

T_BKT = 0.8929  # accepted value, J = 1 (Hasenbusch 2005)


def universal(T):
    """The Nelson-Kosterlitz line that Y(T) crosses at T*."""
    return 2 * T / np.pi


def measure(L, T, n_therm, n_meas, seed, start="ordered", n_blocks=20):
    """One (L, T) run. Top level so ProcessPoolExecutor can pickle it."""
    rng = np.random.default_rng(seed)
    N = L * L
    rec = run(L, T, n_therm, n_meas, rng, start)
    bl = blocks(rec, n_blocks)
    Y, dY = mean_err([stiffness(b, T, N) for b in bl])
    E, dE = mean_err([b["E"].mean() / N for b in bl])
    nv, dnv = mean_err([b["n_vort"].mean() / N for b in bl])
    assert np.all(rec["q"] == 0), "net vorticity must vanish on a torus"
    return dict(L=L, T=T, start=start, Y=Y, dY=dY, E=E, dE=dE,
                n_vort=nv, dn_vort=dnv, theta=rec["theta"], delta=rec["delta"])


def crossing(Ts, Y):
    """First T where Y(T) falls through 2 T / pi, by linear interpolation.

    f = Y - 2T/pi starts positive (stiff) and ends negative (free vortices);
    the crossing is bracketed by the first sign change.
    """
    f = np.asarray(Y) - universal(np.asarray(Ts))
    assert f[0] > 0 and f[-1] < 0, "the T range must bracket the crossing"
    k = np.flatnonzero((f[:-1] > 0) & (f[1:] < 0))[0]
    return Ts[k] + (Ts[k + 1] - Ts[k]) * f[k] / (f[k] - f[k + 1])


def crossing_err(Ts, Y, dY, rng, n_resample=2000):
    """T* and its error, from resampling each Y(T) within its own error bar.

    The runs at different T are independent, so each Y_i is redrawn on its own.
    """
    Ts = np.asarray(Ts)
    draws = [crossing(Ts, rng.normal(Y, dY)) for _ in range(n_resample)]
    return crossing(Ts, Y), np.std(draws, ddof=1)


def fit_log(Ls, Tstar, dTstar):
    """Weighted straight-line fit of T* against 1 / (ln L)^2.

    Returns (T_BKT estimate, its error, slope c, its error).
    """
    x = 1 / np.log(Ls) ** 2
    (c, T0), cov = np.polyfit(x, Tstar, 1, w=1 / np.asarray(dTstar), cov=True)
    return T0, np.sqrt(cov[1, 1]), c, np.sqrt(cov[0, 0])
