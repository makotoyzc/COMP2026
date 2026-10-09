"""Step 3: swap the rectangle mask for a stadium mask, and measure what it costs.

The rectangle's walls sat exactly on grid lines, so the mask was the exact shape
and the only error was the stencil's own h^2. A stadium has a curved wall, which
no square grid can follow, so the mask approximates the arc by a staircase. This
file measures how much that hurts.

Geometry: a quarter of a Bunimovich stadium, flat-wall half-length a = 1 and cap
radius r = 1, with hard walls on all four sides including the two symmetry axes.
Quarter and not the whole stadium because the full table has two mirror
symmetries, so its spectrum is four independent spectra laid on top of each
other, which ruins the level statistics later.

  1  shape.   Counting unknowns gives the mask's area, and counting dropped
     stencil links gives its perimeter. Compare both to the real quarter stadium.
  2  cost.    Fit E_n(h) = E_n(0) + c h^p over 19 grids and compare the result
     against the rectangle at the same spacing.
  3  Weyl.    The number of levels below E should be
     N(E) = A E / 4pi - P sqrt(E) / 4pi for hard walls. The area term is easy to
     satisfy; the perimeter term only comes out right if the mask really is this
     shape. Where the count drifts away from N(E) is where the grid stops
     resolving the wavelength, which sets how far up the spectrum we can trust.

Run:  python checks_stadium.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import billiard as bq

A_EXACT = 1.0 + np.pi / 4             # a*r + pi r^2 / 4 with a = r = 1
P_EXACT = 3.0 + np.pi / 2             # three straight sides of length 1, plus pi r/2
HS = [1 / 40, 1 / 60, 1 / 80, 1 / 120]


def fit_power(h, y):
    """Least squares fit of y = y0 + c h^p.

    Only p is nonlinear, so scan p on a grid and solve for y0 and c exactly at
    each value. Handing all three parameters to a general optimiser instead lets
    it run away to c h^63, which is what a three-parameter fit of a gently
    curving sequence will do if nothing stops it.
    """
    best = None
    for q in np.arange(0.4, 2.51, 0.005):
        M = np.column_stack([np.ones_like(h), h ** q])
        coef = np.linalg.lstsq(M, y, rcond=None)[0]
        r = float(np.sum((M @ coef - y) ** 2))
        if best is None or r < best[0]:
            best = (r, coef[0], coef[1], q)
    return best[1], best[2], best[3]


def check_1_shape():
    print("1  does the mask have the right shape?")
    print(f"     true quarter stadium:  area {A_EXACT:.6f}   perimeter {P_EXACT:.6f}")
    for h in HS:
        mask, _, _ = bq.quarter_stadium_mask(h)
        pad = np.zeros(np.array(mask.shape) + 2, dtype=int)
        pad[1:-1, 1:-1] = mask
        nb = pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]
        drop = np.where(mask, 4 - nb, 0)
        area, perim = mask.sum() * h ** 2, drop.sum() * h
        print(f"     h = 1/{round(1/h):3d}  {mask.sum():6d} unknowns   "
              f"area {area:.6f} ({area/A_EXACT-1:+.2%})   "
              f"perimeter {perim:.6f} ({perim/P_EXACT-1:+.2%})")
    print("     Area converges. Perimeter does not: it stays about 30% too long\n"
          "     at every h, because a staircase climbing a 45-degree slope is\n"
          "     always sqrt(2) times longer than the slope, however fine the\n"
          "     steps. That is a property of the staircase, not an error that\n"
          "     shrinks, so do not use the counted perimeter for anything.\n")


def check_2_cost():
    print("2  what the staircase costs")
    ns = np.arange(30, 211, 10)
    h = 1.0 / ns
    E = np.zeros((len(ns), 25))
    deficit = np.zeros(len(ns))
    for i, n in enumerate(ns):
        mask, _, _ = bq.quarter_stadium_mask(1.0 / n)
        E[i], _ = bq.solve(mask, 1.0 / n, 25)
        deficit[i] = A_EXACT - mask.sum() / n ** 2

    y0, ca, pa = fit_power(h, deficit)
    print(f"     area the mask misses:  {y0:+.4f} + {ca:.3f} h^{pa:.3f}   "
          f"-> proportional to h")
    assert 0.85 < pa < 1.15

    print("     eigenvalues, fitted as E_n(h) = E_n(0) + c h^p over 19 grids:")
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
        print(f"       E_{n+1:<3d} E(1/30) = {E[0, n]:9.4f}   E(1/210) = {E[-1, n]:9.4f}"
              f"   E(0) = {E0:9.4f}   p = {q:.2f}"
              f"   (fit rms {rmsp:.3f}; forcing p = 1 gives {rms1:.3f})")
        assert 0.80 < q < 1.80

    print("     p is not one number. It is 0.97 for the ground state and about\n"
          "     1.6 by level 20, and forcing p = 1 fits the high levels three\n"
          "     times worse, so the error is a mixture of powers rather than a\n"
          "     single clean rate.\n"
          "     The size of the error is smaller than the area loss suggests. If\n"
          "     E simply scaled as 1/A, the ground state would need a prefactor\n"
          f"     of {-ref[0]*ca/A_EXACT:.1f}; the fit gives about a sixth of that. The reason is\n"
          "     that the area the staircase removes is a thin strip along the\n"
          "     wall, where |psi|^2 is already close to zero.\n")

    print("     relative error of level 20, rectangle vs stadium at the same h:")
    rect = {}
    for f in (1, 2, 3):
        nx, ny, hr = 99 * f, 70 * f, 1.0 / (70 * f)
        mask, _, _ = bq.rect_mask(nx, ny, hr)
        Er, _ = bq.solve(mask, hr, 20)
        Ec = bq.exact_rect_continuum(nx, ny, hr, 20)
        rect[hr] = abs(Er[19] - Ec[19]) / Ec[19]
        near = np.argmin(abs(h - hr))
        stad = abs(E[near, 19] - ref[19]) / ref[19]
        print(f"       h = 1/{round(1/hr):3d}   rectangle {rect[hr]:.2e}   "
              f"stadium {stad:.2e}   stadium is {stad/rect[hr]:.1f}x worse")
    print("     So the curved wall costs a factor of two or three in accuracy,\n"
          "     and that factor stays flat as h shrinks rather than growing.\n"
          "     The bigger loss is in the first column: the rectangle's error is\n"
          "     measured against a formula, while every stadium error here had to\n"
          "     be measured against E(0) taken from a fit to the same data. The\n"
          "     curved wall costs less accuracy than it costs certainty, and that\n"
          "     is why step 2 is done on a rectangle.\n"
          "     Either way it is good enough for what follows, which needs the\n"
          "     right wavefunctions and the right spacings, not six digits of any\n"
          "     one level.\n")

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for n, m, lab in ((0, 'o', "stadium, level 1"), (4, 's', "stadium, level 5"),
                      (19, '^', "stadium, level 20")):
        ax.loglog(h, abs(E[:, n] - ref[n]) / ref[n], m + '-', ms=4, lw=1, label=lab)
    hr = np.array(sorted(rect))
    ax.loglog(hr, [rect[x] for x in hr], 'k*--', ms=11,
              label="rectangle, level 20\n(error known exactly)")
    h0, y0_ = h[0], abs(E[0, 19] - ref[19]) / ref[19]
    for p_, lab in ((1.0, "slope 1"), (2.0, "slope 2")):
        ax.loglog(h, y0_ * (h / h0) ** p_, ':', color='0.55', lw=1)
        ax.text(h[-1] * 0.97, y0_ * (h[-1] / h0) ** p_, lab, color='0.45',
                fontsize=8, ha='left', va='center')
    ax.set_xlim(h[-1] * 0.7, h[0] * 1.15)
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("relative error in the eigenvalue")
    ax.set_title("A curved wall costs 2-3x accuracy, and costs the exact reference")
    ax.legend(fontsize=8, loc='lower right')
    ax.text(0.03, 0.97, "The stadium points are measured against a fitted E(0),\n"
            "so their scatter is partly the fit's. The rectangle points\n"
            "are measured against a closed-form answer.",
            transform=ax.transAxes, fontsize=7.5, color='0.35', va='top')
    fig.tight_layout()
    fig.savefig("figures/staircase_cost.png", dpi=150)
    print("     wrote figures/staircase_cost.png\n")


def check_3_weyl():
    h, k = 1 / 120, 400
    mask, _, _ = bq.quarter_stadium_mask(h)
    E, _ = bq.solve(mask, h, k)
    print(f"3  Weyl's law at h = 1/120, {k} levels, {mask.sum()} unknowns")
    N = A_EXACT * E / (4 * np.pi) - P_EXACT * np.sqrt(E) / (4 * np.pi)
    resid = np.arange(1, k + 1) - N
    print("     level      E      N(E) predicted   difference   wavelength")
    for i in (0, 24, 99, 199, 299, 399):
        print(f"     {i+1:5d}  {E[i]:9.1f}   {N[i]:12.2f}   {resid[i]:+10.2f}   "
              f"{2*np.pi/np.sqrt(E[i])/h:7.1f} h")
    print(f"     worst difference over the first 100 levels: "
          f"{abs(resid[:100]).max():.2f} levels")
    assert abs(resid[:100]).max() < 4.0
    print("     The count follows Weyl's law with both terms, so the mask has the\n"
          "     right area and the right perimeter. It starts drifting once the\n"
          "     wavelength drops below roughly 10 grid spacings, which is the\n"
          "     practical ceiling on how high up the spectrum to go.\n")


if __name__ == "__main__":
    check_1_shape()
    check_2_cost()
    check_3_weyl()
    print("all three checks passed")
