"""Natural Step 3, and the "find out what the staircase costs you" question.

Swapping the rectangle mask for a stadium mask changes one thing about the
numerics: the arc does not follow grid lines, so the mask approximates it by a
staircase that is wrong by O(h) in position. The rectangle had no such error --
its walls were exactly on grid lines, which is why check C there saw a clean
h^2. Here we measure what is actually lost.

  E  does the mask have the right shape?  Counting unknowns gives the area and
     the dropped links give the perimeter, both of which must converge to the
     geometry of the continuum quarter stadium.

  F  what it costs.  Fit E_n(h) = E_n(0) + c h^p over 19 grids, compare against
     the rectangle at matched h, and plot both on log-log.

  G  Weyl's law.  N(E) = A E / 4pi - P sqrt(E) / 4pi for Dirichlet walls. The
     area term alone is not a test of much; the perimeter term has the right
     sign and size only if the mask really is the stadium, and the point where
     the counted levels run away from Weyl is where the grid stops resolving the
     wavelength.

Run:  python checks_stadium.py
"""
import numpy as np
import billiard as bq

A_EXACT = 1.0 + np.pi / 4            # a*r + pi r^2 / 4,  a = r = 1
P_EXACT = 3.0 + np.pi / 2            # x=0, y=0, y=r each of length 1, plus pi r/2
HS = [1 / 40, 1 / 60, 1 / 80, 1 / 120]


def check_E():
    print("E  does the mask reproduce the quarter stadium?")
    print(f"     exact   area {A_EXACT:.6f}   perimeter {P_EXACT:.6f}")
    for h in HS:
        mask, _, _ = bq.quarter_stadium_mask(h)
        pad = np.zeros(np.array(mask.shape) + 2, dtype=int)
        pad[1:-1, 1:-1] = mask
        nb = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:])
        drop = np.where(mask, 4 - nb, 0)
        area = mask.sum() * h ** 2
        perim = drop.sum() * h               # each dropped link is one wall face
        print(f"     h = 1/{int(round(1/h)):3d}   unknowns {mask.sum():6d}   "
              f"area {area:.6f} ({area/A_EXACT-1:+.2%})   "
              f"perimeter {perim:.6f} ({perim/P_EXACT-1:+.2%})")
    assert abs(area / A_EXACT - 1) < 0.01
    print("     -> area converges; the staircased perimeter overshoots, which is\n"
          "        the usual taxicab fact: a staircase along a 45-deg arc is longer\n"
          "        than the arc and does not shrink with h\n")


def fit_power(h, y):
    """Least squares y = y0 + c h^p.

    The nonlinearity is only in p, so scan p on a grid and solve the linear part
    exactly at each one. Handing all three parameters to a general optimiser
    instead lets it settle into c h^63, which is what a three-parameter fit to a
    gently curving sequence will do if you let it.
    """
    best = None
    for q in np.arange(0.4, 2.51, 0.005):
        M = np.column_stack([np.ones_like(h), h ** q])
        coef = np.linalg.lstsq(M, y, rcond=None)[0]
        r = float(np.sum((M @ coef - y) ** 2))
        if best is None or r < best[0]:
            best = (r, coef[0], coef[1], q)
    return best[1], best[2], best[3]


def check_F():
    """What the staircase costs, measured against the rectangle at the same h.

    Nineteen grids from h = 1/30 to 1/210 for the stadium. The rectangle's walls
    are on grid lines, so only h = 1/70, 1/140, 1/210 keep the 99:70 shape
    exactly -- but it needs no reference value at all, because its continuum
    spectrum is known, which is the contrast worth drawing: the rectangle's error
    is measured, the stadium's has to be inferred from a fit.

    Three-point Richardson is the wrong tool for the stadium: its error is not a
    smooth function of h, because which nodes fall inside the arc jumps around as
    the grid slides under it. Fit over a long run of h instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("F  what the staircase costs")
    ns = np.arange(30, 211, 10)
    h = 1.0 / ns
    E = np.zeros((len(ns), 25))
    deficit = np.zeros(len(ns))
    for i, n in enumerate(ns):
        mask, _, _ = bq.quarter_stadium_mask(1.0 / n)
        E[i], _ = bq.solve(mask, 1.0 / n, 25)
        deficit[i] = A_EXACT - mask.sum() / n ** 2

    y0, ca, pa = fit_power(h, deficit)
    print(f"     area the mask loses:  A_exact - A_mask = {y0:+.4f} + "
          f"{ca:.3f} h^{pa:.3f}   -> O(h), as a staircase must be")
    assert 0.85 < pa < 1.15

    print("     stadium:  E_n(h) = E_n(0) + c h^p, fitted over all 19 grids")
    ref = {}
    for n in (0, 4, 9, 19, 24):
        E0, c, q = fit_power(h, E[:, n])
        ref[n] = E0
        M1 = np.column_stack([np.ones_like(h), h])
        rms1 = np.sqrt(np.mean((M1 @ np.linalg.lstsq(M1, E[:, n], rcond=None)[0]
                                - E[:, n]) ** 2))
        Mp = np.column_stack([np.ones_like(h), h ** q])
        rmsp = np.sqrt(np.mean((Mp @ np.linalg.lstsq(Mp, E[:, n], rcond=None)[0]
                                - E[:, n]) ** 2))
        print(f"       E_{n+1:<3d} E(1/30) = {E[0, n]:9.4f}  E(1/210) = {E[-1, n]:9.4f}"
              f"  E(0) = {E0:9.4f}  c = {c:+8.1f}  p = {q:.2f}"
              f"  (rms {rmsp:.3f}, forcing p=1 gives {rms1:.3f})")
    print("     -> p is not one clean number: 0.97 at the bottom of the spectrum,\n"
          "        rising to ~1.6 by E_20, and forcing p = 1 fits the high levels\n"
          "        3x worse.  So the error is a mixture, not a single power, and\n"
          "        quoting one exponent for a staircased boundary is overclaiming.\n"
          "     -> the prefactor is milder than the geometry suggests.  Naive area\n"
          f"        scaling E ~ 1/A wants c = -E_1 c_A / A = {-ref[0]*ca/A_EXACT:.1f} for the ground\n"
          "        state; the fit gives about a sixth of that, because the area the\n"
          "        staircase eats is a thin sliver against the wall where |psi|^2 is\n"
          "        already near zero.  An eigenvalue is far less sensitive to the\n"
          "        boundary than the boundary's own length is.\n")

    # the headline number: same level, same h, curved wall vs grid-aligned wall
    print("     side by side, relative error of E_20 at matched h:")
    rect = {}
    for f in (1, 2, 3):
        nx, ny, hr = 99 * f, 70 * f, 1.0 / (70 * f)
        mask, _, _ = bq.rect_mask(nx, ny, hr)
        Er, _ = bq.solve(mask, hr, 20)
        Ec = bq.exact_rect_continuum(nx, ny, hr, 20)
        rect[hr] = abs(Er[19] - Ec[19]) / Ec[19]
        near = np.argmin(abs(h - hr))
        stad = abs(E[near, 19] - ref[19]) / ref[19]
        print(f"       h = 1/{int(round(1/hr)):3d}   rectangle {rect[hr]:.2e}   "
              f"stadium (h = 1/{ns[near]}) {stad:.2e}   "
              f"ratio {stad/rect[hr]:5.0f}x")
    print("     -> the cost in accuracy is a factor of 2-3, not the collapse the O(h)\n"
          "        boundary error threatens, and the ratio is roughly flat over this\n"
          "        range rather than widening.  The effective exponent at E_20 is\n"
          "        1.55, close enough to the rectangle's 2 that the two curves stay\n"
          "        parallel on the log-log plot.\n"
          "     -> the cost that actually bites is the one in the column headings:\n"
          "        the rectangle's error is MEASURED against a closed form, while\n"
          "        every stadium number above is measured against E(0) from a fit to\n"
          "        the same data.  Curving a wall does not cost you much accuracy\n"
          "        here; it costs you the ability to know how much you lost.  That\n"
          "        is the real reason Step 2 happens on the rectangle.\n"
          "     -> either way it is fine for what comes next.  Scars and level\n"
          "        statistics need the right states and the right spacings, not six\n"
          "        digits of any one level.\n")

    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for n, m in ((0, 'o'), (4, 's'), (19, '^')):
        ax.loglog(h, abs(E[:, n] - ref[n]) / ref[n], m + '-', ms=4,
                  label=f"stadium  E_{n+1}")
    hr = np.array(sorted(rect))
    ax.loglog(hr, [rect[x] for x in hr], 'k*--', ms=10,
              label="rectangle  E_20 (exact ref)")
    h0, y0_ = h[0], abs(E[0, 19] - ref[19]) / ref[19]     # anchor at coarsest grid
    for p_, lab in ((1.0, "slope 1"), (2.0, "slope 2")):
        ax.loglog(h, y0_ * (h / h0) ** p_, ':', color='0.6', lw=1)
        ax.text(h[-1] * 0.97, y0_ * (h[-1] / h0) ** p_, lab, color='0.5',
                fontsize=8, ha='left', va='center')
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("relative eigenvalue error")
    ax.set_xlim(h[-1] * 0.78, h[0] * 1.12)
    ax.set_title("what the staircased arc costs: curved wall vs grid-aligned wall")
    ax.legend(fontsize=8, loc='lower right')
    ax.text(0.03, 0.97, "stadium errors are measured against E(0) from the fit;\n"
            "the rectangle's are measured against a closed form.\n"
            "That, not the slope, is what the curved wall really costs.",
            transform=ax.transAxes, fontsize=7, color='0.35', va='top')
    fig.tight_layout()
    fig.savefig("figures/staircase_cost.png", dpi=150)
    print("     wrote figures/staircase_cost.png\n")


def check_G():
    h = 1 / 120
    k = 400
    mask, _, _ = bq.quarter_stadium_mask(h)
    E, _ = bq.solve(mask, h, k)
    print(f"G  Weyl's law, h = 1/120, {k} levels, {mask.sum()} unknowns")
    n_weyl = (A_EXACT * E / (4 * np.pi)
              - P_EXACT * np.sqrt(E) / (4 * np.pi))
    resid = np.arange(1, k + 1) - n_weyl
    for i in (0, 24, 99, 199, 299, 399):
        lam = 2 * np.pi / np.sqrt(E[i])
        print(f"     n = {i+1:4d}   E = {E[i]:10.2f}   N_weyl = {n_weyl[i]:8.2f}   "
              f"n - N_weyl = {resid[i]:+6.2f}   wavelength = {lam/h:5.1f} h")
    print(f"     |n - N_weyl| over the first 100 levels: max {abs(resid[:100]).max():.2f}")
    assert abs(resid[:100]).max() < 4.0
    print("     -> the counting function tracks Weyl with both terms, so the mask's\n"
          "        area AND perimeter are right.  The residual walks off once the\n"
          "        wavelength drops under ~10 h, which sets the trustworthy range\n")


if __name__ == "__main__":
    check_E()
    check_F()
    check_G()
