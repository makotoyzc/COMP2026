"""From pairing amplitude to phase stiffness: the missing half of a 2D transition.

THE TWO HALVES
--------------
A superconducting order parameter is a complex number, Delta exp(i theta). Mean
field theory computes its size and assumes its phase is rigid. That is exactly
the structure of Ising mean field theory: one scalar order parameter fixed by
its own average field,

    Delta = U <c_down c_up>        versus        m = tanh(beta J z m)

and both vanish at a mean-field transition temperature. In 2D the assumption
behind that calculation is false. A continuous phase cannot order in 2D
(Mermin-Wagner), and what actually kills the superconductor is the phase
unbinding its vortices -- the XY problem in the rest of this folder.

So the two halves fit together. This module does the amplitude half, on the
lattice, and converts its answer into the one number the XY half needs: the
coupling J.

AMPLITUDE: the lattice gap equation
-----------------------------------
Nearest-neighbor square lattice, on-site attraction U, s-wave gap:

    xi_k = -2 t (cos kx + cos ky) - mu,     E_k = sqrt(xi_k^2 + Delta^2)

    1 = (U / N) sum_k tanh(E_k / 2T) / (2 E_k)

No Debye cutoff is needed -- the band width supplies one.

PHASE: the dirty-limit stiffness
--------------------------------
For a 2D film of normal-state sheet resistance R_sq, the superfluid stiffness
follows from the gap (the standard dirty-limit result behind Beasley, Mooij and
Orlando 1979):

    J(T) = (Delta(T) / 2) (R_Q / R_sq) tanh(Delta(T) / 2T),   R_Q = h / 4e^2

R_Q = 6.45 kOhm. Only the ratio R_sq / R_Q enters, so the whole amplitude-to-
phase bridge has one knob, and it is a measurable one.

TRANSITION: two criteria, and the gap between them is the point
--------------------------------------------------------------
    bare Nelson-Kosterlitz:   T = (pi / 2) J(T)       -> coefficient 1.5708
    lattice XY, measured:     T = 0.8929 J(T)         -> from step 2

The first ignores vortices entirely: it asks when the UNrenormalized stiffness
can no longer pay for a free vortex. The second is what the simulation actually
finds once vortices and spin waves have eaten into the stiffness. Their ratio,
0.568, is the size of the correction mean field theory cannot see.

That 0.8929 belongs to the nearest-neighbor XY model, whose vortex core energy
is its own. A real film's core energy differs, so treat the lattice number as
the SIZE of the vortex correction, not as a constant to carry into data.
"""
import numpy as np

from jump import T_BKT

NK = np.pi / 2       # bare Nelson-Kosterlitz coefficient, no vortex renormalization
LATTICE = T_BKT      # 0.8929, the same coefficient measured in step 2


def xi(t, mu, nk):
    """Tight-binding energies on an nk x nk k-grid, flattened."""
    k = 2 * np.pi * np.arange(nk) / nk
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return (-2 * t * (np.cos(kx) + np.cos(ky)) - mu).ravel()


def gap_rhs(Delta, T, U, xi_k):
    """The right-hand side of the gap equation; it equals 1 at the solution."""
    E = np.hypot(xi_k, Delta)
    return U * np.mean(np.tanh(E / (2 * T)) / (2 * E))


def solve_gap(T, U, xi_k, hi=10.0, tol=1e-12):
    """Delta(T) by bisection. Returns 0 when the only solution is the normal state.

    gap_rhs falls monotonically with Delta, so if it is already below 1 at
    Delta = 0 there is no superconducting root.
    """
    if gap_rhs(0.0, T, U, xi_k) < 1:
        return 0.0
    lo = 0.0
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap_rhs(mid, T, U, xi_k) > 1 else (lo, mid)
    return 0.5 * (lo + hi)


def Tc_mf(U, xi_k, lo=1e-4, hi=2.0, tol=1e-10):
    """Mean-field Tc: where the Delta = 0 gap equation stops having a solution."""
    assert gap_rhs(0.0, lo, U, xi_k) > 1 > gap_rhs(0.0, hi, U, xi_k), "bracket Tc"
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap_rhs(0.0, mid, U, xi_k) > 1 else (lo, mid)
    return 0.5 * (lo + hi)


def stiffness(Delta, T, R_ratio):
    """Dirty-limit phase stiffness J(T), in the same energy units as Delta."""
    return 0.5 * Delta / R_ratio * np.tanh(np.divide(Delta, 2 * T))


def gap_curve(U, xi_k, Tc, n=400):
    """Delta(T) on a grid from Tc down to zero. Independent of R_sq, so it is
    solved once and reused for every sheet resistance."""
    Ts = np.linspace(Tc, Tc / n, n)
    return Ts, np.array([solve_gap(T, U, xi_k) for T in Ts])


def T_bkt(Ts, Deltas, R_ratio, coeff):
    """Where T = coeff * J(T), by walking down from Tc and interpolating.

    J vanishes at Tc and grows as T falls, so T - coeff * J(T) starts positive
    and the first sign change coming down is the transition. If it never
    changes sign the stiffness never catches up and there is no phase-coherent
    state at any temperature.
    """
    f = Ts - coeff * stiffness(Deltas, Ts, R_ratio)
    assert f[0] > 0, "J must vanish at Tc"
    if f[-1] > 0:
        return 0.0
    k = np.flatnonzero(f < 0)[0]
    return Ts[k - 1] + (Ts[k] - Ts[k - 1]) * f[k - 1] / (f[k - 1] - f[k])
